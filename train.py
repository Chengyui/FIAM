import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _,_,_,_ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            problem,
            tb_logger,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        problem,
        tb_logger,
        opts
):

    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)

    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood,ll_list,pi,_ = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    reinforce_seperate_loss,adv_costs = calculate_seperate_policy_gradient(x, pi, ll_list, problem, cost, opts)
    loss = reinforce_loss + bl_loss + reinforce_seperate_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts,reinforce_seperate_loss,adv_costs)


def pre_process_seperate_policy_gradient(input,trace,ll_list,problem,origin_cost,opts,gamma=0.5):
    return 0
def calculate_seperate_policy_gradient(input,trace,ll_list,problem,origin_cost,opts,gamma=0.5):
    """
    input:
    input:{batch_size,num_node,2}
    trace:{batch_size,trace_length} for tsp trace_length is equal to num_node
    ll_list:{batch_size,trace}
    problem:tsp,cvrp....
    origin_cost:{batch_size,1}

    output:
    reward: {batch_size,trace_length}
    """

    # def get_costs2(input,trace):
    #     """
    #     input (512,20,2)
    #     trace (512,x,20)
    #     """
    #     batch_size = trace.size(0)
    #     trace_length = trace.size(2)
    #     trace = trace.view(-1, trace.size(2)) # view后，trace的1-20是对应同一个instance
    #     input = input.repeat(1,trace.size(1),1) # 而input进行repeat，1-20对应的是1-20个instance
    #     input = input.view(-1,trace_length,input.size(2))
    #
    #     d = input.gather(1, trace.unsqueeze(-1).expand_as(input))
    #     cost = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)
    #     cost = cost.view(batch_size,-1)
    #     return cost
    if opts.problem =='tsp':
        adv = torch.zeros(input.size(0),input.size(1)).to(opts.device)
        # 对于每一个被交换的节点，最好的cost
        best_costs = torch.full((trace.size(0),trace.size(1)), float('inf')).to(opts.device)
        # best_costs1 = torch.full((trace.size(0), trace.size(1)), float('inf')).to(opts.device)

        # for i in range(trace.size(1)):
        #     for j in range(i,trace.size(1)):
        #         trace_ = trace.clone()
        #         trace_[:,[i,j]] = trace_[:,[j,i]]
        #         # shape:{batch_size,1}
        #         cost_swap = get_costs(input,trace_)
        #         best_costs[:, i] = torch.where(best_costs[:, i] > cost_swap, cost_swap, best_costs[:, i])
        #         best_costs[:, j] = torch.where(best_costs[:, j] > cost_swap, cost_swap, best_costs[:, j])


    #bug解决了一个，为什么best_cost和best_cost1有差距，是因为旧的循环我故意不计算他自己不动的那部分，而新的循环我给搞忘了，计算了不动的那部分，导致了他们两个的不同
        batch_size = trace.size(0)
        trace_length = trace.size(1)

        for i in range(trace.size(1)):
            trace_all = trace.clone().unsqueeze(1).repeat(1, trace.size(1), 1)
            pair = torch.zeros(trace.size(1),2,dtype=int).to(opts.device)
            pair[:,0] = i
            pair[:,1] = torch.arange(trace.size(1))
            pair_ = pair.clone()
            pair_[:,[0,1]] = pair_[:,[1,0]]
            sub_tensor = torch.gather(trace_all,dim=2,index=pair_.unsqueeze(0).expand(trace.size(0),-1,-1))
            trace_all.scatter_(dim=2, index=pair.unsqueeze(0).expand(trace.size(0), -1, -1), src=sub_tensor)
            # trace_all = trace_all.view(-1, trace.size(1))
            #trace_all :{512,20,20},cost_swap:{512,20}
            trace_all = trace_all.view(-1, trace_all.size(2))

            input_ = input.repeat(1, trace_all.size(1), 1)

            input_ = input_.view(-1, trace_length, input_.size(2))

            cost_swap,_ = problem.get_costs(input_, trace_all)

            cost_swap = cost_swap.view(batch_size, -1)

            best_costs[:,i],_ = cost_swap.min(dim=1)
    elif opts.problem == 'cvrp':
        batch_size = trace.size(0)
        trace_length = trace.size(1)
        num_nodes = input['loc'].size(1)

        adv = torch.zeros(trace.size(0), trace_length).to(opts.device)
        # 对于每一个被交换的节点，最好的cost
        best_costs = torch.full((trace.size(0), trace.size(1)), float('inf')).to(opts.device)

        # bug解决了一个，为什么best_cost和best_cost1有差距，是因为旧的循环我故意不计算他自己不动的那部分，而新的循环我给搞忘了，计算了不动的那部分，导致了他们两个的不同


        for i in range(trace.size(1)):
            trace_all = trace.clone().unsqueeze(1).repeat(1, trace_length, 1)
            pair = torch.zeros(trace.size(1), 2, dtype=int).to(opts.device)
            pair[:, 0] = i
            pair[:, 1] = torch.arange(trace.size(1))
            pair_ = pair.clone()
            pair_[:, [0, 1]] = pair_[:, [1, 0]]
            sub_tensor = torch.gather(trace_all, dim=2, index=pair_.unsqueeze(0).expand(trace.size(0), -1, -1))
            trace_all.scatter_(dim=2, index=pair.unsqueeze(0).expand(trace.size(0), -1, -1), src=sub_tensor)
            # trace_all = trace_all.view(-1, trace.size(1))
            # trace_all :{512,20,20},cost_swap:{512,20}
            trace_all = trace_all.view(-1, trace_all.size(2))

            input_ = input.copy()
            input_['loc'] = input_['loc'].repeat(1, trace_length, 1)
            input_['loc'] = input_['loc'].view(-1,num_nodes,input_['loc'].size(2))
            input_['demand'] = input_['demand'].repeat(1,trace_length)
            input_['demand'] = input_['demand'].view(-1,num_nodes)
            input_['depot'] = input_['depot'].repeat(1,trace_length)
            input_['depot'] = input_['depot'].view(-1,2)

            cost_swap, _ = problem.get_costs(input_, trace_all)

            cost_swap = cost_swap.view(batch_size, -1)

            best_costs[:, i], _ = cost_swap.min(dim=1)

    adv = origin_cost.unsqueeze(-1).repeat(1,trace.size(1)) - best_costs
    # adv[adv<0] = 0
    G = 0
    loss = 0
    for i in range(trace.size(1)):
        G = gamma*G + adv[:,i]
        loss = loss+(G*ll_list[:,i]).mean()


    return loss,adv
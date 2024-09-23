import torch
import tqdm
from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import constrainScoreByWhole
from torch.utils.tensorboard import SummaryWriter

# from tensorboardX import SummaryWriter
import shutil
import numpy as np
import builtins as __builtin__
from args import VerboseMode
import math

def print(*args, **kwargs):
    if VerboseMode:
        # __builtin__.print('My overridden print() function!')
        return __builtin__.print(*args, **kwargs)

writer = SummaryWriter()
__all__ = ["train", "validate", "modifier"]


def calculateGrad(model, fn_avg, fn_list, args):
    for n, m in model.named_modules():
        if hasattr(m, "scores") and m.prune:
            m.scores.grad.data += 1 / (args.K - 1) * (fn_list[0] - fn_avg) * getattr(
                m, "stored_mask_0"
            ) + 1 / (args.K - 1) * (fn_list[1] - fn_avg) * getattr(m, "stored_mask_1")
            if "IMP" in args.conv_type:
                # print("process grad in another way")
                m.scores.grad.data = torch.where(
                    m.scores > 0.9,
                    m.scores.grad.data,
                    m.scores.grad.data * args.scaling_para,
                )


def calculateGrad_pge(model, fn_avg, fn_list, args):
    for n, m in model.named_modules():
        if hasattr(m, "scores") and m.prune:
            m.scores.grad.data += 1 / args.K * (
                fn_list[0] * getattr(m, "stored_mask_0")
            ) + 1 / args.K * (fn_list[1] * getattr(m, "stored_mask_1"))
            if "IMP" in args.conv_type:
                m.scores.grad.data = torch.where(
                    m.scores > 0.9,
                    m.scores.grad.data,
                    m.scores.grad.data * args.scaling_para,
                )


def calScalingPara(model, args):
    remaining_part = 0
    original_part = 0
    for n, m in model.named_modules():
        if hasattr(m, "scores") and m.prune:
            original_part += (torch.lt(m.scores, 0.9).float() * m.scores).sum().item()
            ge_loc = torch.ge(m.scores, 0.9).float()
            remaining_part += (m.scores.sum() - ge_loc.sum()).item()
    if remaining_part < 0:
        # print("remaining_part negative", remaining_part)
        remaining_part = 0
    if original_part == 0:
        args.scaling_para = 0
    else:
        args.scaling_para = remaining_part / original_part


def get_a_batch_data(dataset, batch_offset, batch_size):
    train_xb = torch.zeros(batch_size, 3, 64 ,32)
    train_yb = torch.zeros(batch_size, 1)
    train_gb = torch.zeros(batch_size, 1)
    train_cb = torch.zeros(batch_size, 1)
    for i in range(batch_size):
        (dx, dy, dg, dc) = dataset[batch_offset + i]
        train_xb[i] = dx
        train_yb[i][0] = torch.from_numpy(dy)
        train_gb[i][0] = torch.from_numpy(dg)
        train_cb[i][0] = torch.from_numpy(dc)
    return (train_xb, train_yb, train_gb, train_cb)

def train(
    train_loader, model, ebd, criterion, optimizer, epoch, args, writer, weight_opt
):
    if VerboseMode:
        loss_meter = AverageMeter("Loss", ":.3f")
        train_nll_meter = AverageMeter("train_nll", ":6.2f")
        train_penalty_meter = AverageMeter("train_penalty", ":6.2f")
        weight_norm_meter = AverageMeter("weight_norm", ":6.2f")
        train_acc_meter = AverageMeter("train_acc", ":6.2f")
        train_minacc_meter = AverageMeter("train_minacc", ":6.2f")
        train_majacc_meter = AverageMeter("train_majacc", ":6.2f")
        train_corr_meter = AverageMeter("train_corr", ":6.2f")
        v_meter = AverageMeter("v", ":6.4f")
        max_score_meter = AverageMeter("max_score", ":6.4f")
        l1_meter = AverageMeter("l1", ":6.4f")
        zero_count_meter = AverageMeter("zero_count", ":6.4f")

        l = [
            loss_meter,
            train_nll_meter,
            train_penalty_meter,
            weight_norm_meter,
            train_acc_meter,
            train_minacc_meter,
            train_majacc_meter,
            train_corr_meter,
            l1_meter,
            zero_count_meter,
        ]
        progress = ProgressMeter(
            len(train_loader),
            l,
            prefix=f"Epoch: [{epoch}]",
        )
    else:
        train_acc_meter = AverageMeter("train_acc", ":6.2f")
        train_minacc_meter = AverageMeter("train_minacc", ":6.2f")
        train_majacc_meter = AverageMeter("train_majacc", ":6.2f")
        train_corr_meter = AverageMeter("train_corr", ":6.2f")
        v_meter = AverageMeter("v", ":6.4f")
        max_score_meter = AverageMeter("max_score", ":6.4f")

    model.train()
    args.discrete = False
    args.val_loop = False
    # args.num_batches = len(train_loader)

    # if VerboseMode:
    #     BatchCollections = tqdm.tqdm(enumerate(train_loader), ascii=True, total=len(train_loader))
    # else:
    #     BatchCollections = enumerate(train_loader)
    # for i, (train_x, train_y, train_g, train_c) in BatchCollections:

    if args.use_dataloader:
        if VerboseMode:
            BatchCollections = tqdm.tqdm(enumerate(train_loader), ascii=True, total=len(train_loader))
        else:
            BatchCollections = enumerate(train_loader)
        BatchCollectionsList = list(enumerate(BatchCollections))

    istart = 0
    totalBatch = math.ceil(len(train_loader) * 1.0 / args.batch_size)
    for i in range(totalBatch):
        if args.use_dataloader:
            (train_x, train_y, train_g, train_c) = BatchCollectionsList[i][1][1]
        else:
            batch_size = args.batch_size
            if i == totalBatch - 1:
                batch_size = len(train_loader) - i * args.batch_size
            (train_x, train_y, train_g, train_c) = get_a_batch_data(train_loader, istart, batch_size)

        train_x, train_y, train_g, train_c = (
            train_x.cuda(),
            train_y.cuda().float(),
            train_g.cuda(),
            train_c.cuda(),
        )
        # print(train_x.size(), train_y.size())
        train_c_label = (2 * train_y - 1) * train_c - train_y + 1

        l, tn, tp, wn, t_acc, t_min_acc, t_maj_acc, t_corr = 0, 0, 0, 0, 0, 0, 0, 0
        if optimizer is not None:
            optimizer.zero_grad()
        if weight_opt is not None:
            weight_opt.zero_grad()
        fn_list = []

        for j in range(args.K):
            args.j = j
            if args.irm_type == "rex":
                loss_list = []
                train_logits = model(train_x)
                train_nll = 0
                for i in range(int(train_g.max()) + 1):
                    ei = (train_g == i).view(-1)
                    ey = train_y[ei]
                    el = train_logits[ei]
                    enll = criterion(el, ey)
                    train_nll += enll / (train_g.max() + 1)
                    loss_list.append(enll)
                loss_t = torch.stack(loss_list)
                train_penalty = ((loss_t - loss_t.mean()) ** 2).mean()
            elif args.irm_type == "irmv1":
                train_logits = ebd(train_g).view(-1, 1) * model(train_x)
                train_nll = criterion(train_logits, train_y)
                grad = torch.autograd.grad(
                    train_nll * args.envs_num, ebd.parameters(), create_graph=True
                )[0]
                train_penalty = torch.mean(grad**2)
            elif args.irm_type == "irmv1b":
                e1 = (train_g == 0).view(-1).nonzero().view(-1)
                e2 = (train_g == 1).view(-1).nonzero().view(-1)
                e1 = e1[torch.randperm(len(e1))]
                e2 = e2[torch.randperm(len(e2))]
                s1 = torch.cat([e1[::2], e2[::2]])
                s2 = torch.cat([e1[1::2], e2[1::2]])
                train_logits = ebd(train_g).view(-1, 1) * model(train_x)

                train_nll1 = criterion(train_logits[s1], train_y[s1])
                train_nll2 = criterion(train_logits[s2], train_y[s2])
                train_nll = train_nll1 + train_nll2
                grad1 = torch.autograd.grad(
                    train_nll1 * args.envs_num, ebd.parameters(), create_graph=True
                )[0]
                grad2 = torch.autograd.grad(
                    train_nll2 * args.envs_num, ebd.parameters(), create_graph=True
                )[0]
                train_penalty = torch.mean(torch.abs(grad1 * grad2))
            else:
                raise Exception

            train_acc, train_minacc, train_majacc = args.eval_fn(
                train_logits, train_y, train_c
            )
            # t_c = np.corrcoef(torch.cat((torch.sigmoid(train_logits), train_c_label), 1).t().detach().cpu().numpy())[0,1]
            weight_norm = torch.tensor(0.0).cuda()
            for n, m in model.named_modules():
                if hasattr(m, "weight") and m.weight is not None:
                    weight_norm += m.weight.norm().pow(2)
                if hasattr(m, "bias") and m.bias is not None:
                    weight_norm += m.bias.norm().pow(2)
            # for w in model.parameters():
            #     weight_norm += w.norm().pow(2)
            # print(
            #     "args.step args.penalty_anneal_iters",
            #     args.steps,
            #     args.penalty_anneal_iters,
            # )
            penalty_weight = (
                args.penalty_weight if args.steps >= args.penalty_anneal_iters else 0.0
            )
            # print("penalty weights", penalty_weight)

            if args.use_pgd:
                args.l2_regularizer_weight = 0.0

            loss = (
                train_nll
                + args.l2_regularizer_weight * weight_norm
                + penalty_weight * train_penalty
            )
            if penalty_weight > 1.0:
                loss /= 1.0 + penalty_weight
            loss = loss / args.K
            fn_list.append(loss.item() * args.K)
            loss.backward()

            # for n, m in model.named_modules():
            #     if hasattr(m, "scores"):
            #         print("pr grad mean", n, m.scores.grad.mean().item())
            #         print(m.train_weights)
            l = l + loss.item()
            tn = tn + train_nll.item() / args.K
            tp = tp + train_penalty / args.K
            wn = wn + args.l2_regularizer_weight * weight_norm / args.K
            t_acc = t_acc + train_acc.item() / args.K
            t_min_acc = t_min_acc + train_minacc.item() / args.K
            t_maj_acc = t_maj_acc + train_majacc.item() / args.K
            # t_corr = t_corr + t_c.item() / args.K

        fn_avg = l
        if not args.finetuning:
            if "ReinforceLOO" in args.conv_type:
                calculateGrad(model, fn_avg, fn_list, args)
            if args.conv_type == "Reinforce":
                calculateGrad_pge(model, fn_avg, fn_list, args)

        if VerboseMode:
            loss_meter.update(l, train_x.size(0))
            train_nll_meter.update(tn, train_x.size(0))
            train_penalty_meter.update(tp, train_x.size(0))
            weight_norm_meter.update(wn, train_x.size(0))
            train_acc_meter.update(t_acc, train_x.size(0))
            train_minacc_meter.update(t_min_acc, train_x.size(0))
            train_majacc_meter.update(t_maj_acc, train_x.size(0))
            train_corr_meter.update(t_corr, train_x.size(0))
            l1_norm = model.module.fc.weight.norm(p=1)
            l1_meter.update(l1_norm.item(), train_x.size(0))
        else:
            train_acc_meter.update(t_acc, train_x.size(0))
            train_minacc_meter.update(t_min_acc, train_x.size(0))
            train_majacc_meter.update(t_maj_acc, train_x.size(0))
            train_corr_meter.update(t_corr, train_x.size(0))

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 3)

        if optimizer is not None:
            if "Dense" not in args.conv_type and not args.fix_subnet:
                if args.steps >= len(train_loader) * args.epochs * args.ts:
                    # print(
                    #     "args.steps >= len(train_loader)*args.epochs*args.ts",
                    #     args.steps,
                    #     len(train_loader) * args.epochs * args.ts,
                    # )
                    optimizer.step()
            else:
                optimizer.step()
        if weight_opt is not None:
            weight_opt.step()

        # if args.steps == args.pgd_anneal_iters - 100:
        #     # print("l1 at pgd_anneal_iters", l1_norm.item())
        #     proj_up(model.module, args.z)
        #     # with torch.no_grad():
        #         # args.z = l1_norm.item() * args.fraction_z
        #     print("set z to", args.z)

        if args.use_pgd and args.steps > args.pgd_anneal_iters:
            # print("args.step pgd_anneal_iters", args.steps, args.pgd_anneal_iters)
            if args.steps % args.pgd_skip_steps == 0:
                with torch.no_grad():
                    proj_sort(model.module, args.z, args.rho_tolerance)
            # proj(model.module, args.z)

        args.steps += 1
        if "Dense" not in args.conv_type:
            if not args.finetuning:
                with torch.no_grad():  # look into this TODO
                    constrainScoreByWhole(model, v_meter, max_score_meter)
                    if "IMP" in args.conv_type:
                        calScalingPara(model, args)
                        t = len(train_loader) * epoch + i
                        if VerboseMode:
                            writer.add_scalar(
                                f"train/scaling_para", args.scaling_para, global_step=t
                            )
                        # print("scalingpara at this batch", args.scaling_para)
        istart += args.batch_size

    if args.use_pgd and args.steps > args.pgd_anneal_iters:
        print("final projection at end of training")
        with torch.no_grad():
            proj_sort(model.module, args.z, args.rho_tolerance)
    if VerboseMode:
        zero_count_meter.update((model.module.fc.weight == 0).sum().item(), train_x.size(0))
        progress.display(len(train_loader))
        progress.write_to_tensorboard(
            writer, prefix="train" if not args.finetuning else "train_ft", global_step=epoch
    )
    return (
        train_acc_meter.avg,
        train_minacc_meter.avg,
        train_majacc_meter.avg,
        train_corr_meter.avg,
    )


def validate(val_loader, model, criterion, args, writer, epoch):
    if VerboseMode:
        loss_meter = AverageMeter("Loss", ":.3f")
        test_acc_meter = AverageMeter("test_acc", ":6.2f")
        test_minacc_meter = AverageMeter("test_minacc", ":6.2f")
        test_majacc_meter = AverageMeter("test_majacc", ":6.2f")

        loss_meter_d = AverageMeter("Loss_d", ":.3f")
        test_acc_meter_d = AverageMeter("test_acc_d", ":6.2f")
        test_minacc_meter_d = AverageMeter("test_minacc_d", ":6.2f")
        test_majacc_meter_d = AverageMeter("test_majacc_d", ":6.2f")
        test_corr_meter_d = AverageMeter("test_corr_d", ":6.2f")

        progress = ProgressMeter(
            len(val_loader),
            [
                loss_meter,
                test_acc_meter,
                test_minacc_meter,
                test_majacc_meter,
                loss_meter_d,
                test_acc_meter_d,
                test_minacc_meter_d,
                test_majacc_meter_d,
                test_corr_meter_d,
            ],
            prefix="Test: ",
        )
    else:
        loss_meter_d = AverageMeter("Loss_d", ":.3f")
        test_acc_meter_d = AverageMeter("test_acc_d", ":6.2f")
        test_minacc_meter_d = AverageMeter("test_minacc_d", ":6.2f")
        test_majacc_meter_d = AverageMeter("test_majacc_d", ":6.2f")
        test_corr_meter_d = AverageMeter("test_corr_d", ":6.2f")

    args.val_loop = True
    if args.use_running_stats:
        model.eval()
    # if writer is not None:
    #     for n, m in model.named_modules():
    #         if hasattr(m, "scores") and m.prune:
    #             writer.add_histogram(n, m.scores)
    with torch.no_grad():
        # if VerboseMode:
        #     BatchCollections = tqdm.tqdm(enumerate(val_loader), ascii=True, total=len(val_loader))
        # else:
        #     BatchCollections = enumerate(val_loader)
        # for i, (test_x, test_y, test_g, test_c) in BatchCollections:
        if args.use_dataloader:
            if VerboseMode:
                BatchCollections = tqdm.tqdm(enumerate(val_loader), ascii=True, total=len(val_loader))
            else:
                BatchCollections = enumerate(val_loader)
            BatchCollectionsList = list(enumerate(BatchCollections))

        istart = 0
        totalBatch = math.ceil(len(val_loader) * 1.0 / args.batch_size)
        for i in range(totalBatch):
            if args.use_dataloader:
                (test_x, test_y, test_g, test_c) = BatchCollectionsList[i][1][1]
            else:
                batch_size = args.batch_size
                if i == totalBatch - 1:
                    batch_size = len(val_loader) - i * args.batch_size
                (test_x, test_y, test_g, test_c) = get_a_batch_data(val_loader, istart, batch_size)

            test_x, test_y, test_g, test_c = (
                test_x.cuda(),
                test_y.cuda().float(),
                test_g.cuda(),
                test_c.cuda(),
            )
            test_c_label = (2 * test_y - 1) * test_c - test_y + 1
            if VerboseMode:
                args.discrete = False
                test_logits = model(test_x)
                loss = criterion(test_logits, test_y)
                test_acc, test_minacc, test_majacc = args.eval_fn(
                    test_logits, test_y, test_c
                )
                loss_meter.update(loss.item(), test_x.size(0))
                test_acc_meter.update(test_acc.item(), test_x.size(0))
                test_minacc_meter.update(test_minacc.item(), test_x.size(0))
                test_majacc_meter.update(test_majacc.item(), test_x.size(0))

            args.discrete = True
            test_logits_d = model(test_x)
            loss_d = criterion(test_logits_d, test_y)
            test_acc_d, test_minacc_d, test_majacc_d = args.eval_fn(
                test_logits_d, test_y, test_c
            )
            loss_meter_d.update(loss_d.item(), test_x.size(0))
            test_acc_meter_d.update(test_acc_d.item(), test_x.size(0))
            test_minacc_meter_d.update(test_minacc_d.item(), test_x.size(0))
            test_majacc_meter_d.update(test_majacc_d.item(), test_x.size(0))
            test_corr_meter_d.update(
                np.corrcoef(
                    torch.cat((torch.sigmoid(test_logits_d), test_c_label), 1)
                    .t()
                    .detach()
                    .cpu()
                    .numpy()
                )[0, 1]
            )
            if VerboseMode and i % args.print_freq == 0:
                progress.display(i)
            istart += args.batch_size
            # end of loop
        if VerboseMode:
            progress.display(len(val_loader))
            if writer is not None:
                progress.write_to_tensorboard(
                    writer,
                    prefix="test" if not args.finetuning else "test_ft",
                    global_step=epoch,
                )
    return (
        test_acc_meter_d.avg,
        test_minacc_meter_d.avg,
        test_majacc_meter_d.avg,
        loss_meter_d.avg,
        test_corr_meter_d.avg,
    )


def modifier(args, epoch, model):
    return


def proj_up(model, z):
    v = model.fc.weight.data.flatten()
    dim_v = v.shape[0]
    # if torch.norm(v, 1) <= z:
    #     return

    signs = torch.sign(v)
    mu, p = torch.sort(v.abs(), descending=True)
    # signs[p]
    rho = dim_v - 1
    for i in range(dim_v):
        res = mu[i] - (torch.sum(mu[:i]) - z) / (i + 1)
        if res <= 0:
            rho = i - 1
            break

    theta = (torch.sum(mu[:rho]) - z) / (rho + 1)
    trimmed = (mu - theta).clamp(min=0)

    # theta = (torch.sum(mu[:rho]) - z) / (rho + 1)
    # trimmed = (mu - theta).clamp(min=0)

    print("rho", rho)
    # print("rho, theta", rho, theta)
    model.fc.weight.data = (trimmed[p] * signs).reshape(model.fc.weight.shape)
    print("num zeros", (model.fc.weight == 0).sum().item())


def proj_sort(model, z, rho_tolerance):
    v = model.fc.weight.data.flatten()
    dim_v = v.shape[0]
    # if torch.norm(v, 1) <= z:
    #     return

    signs = torch.sign(v)
    mu, p = torch.sort(v.abs(), descending=True)
    # signs[p]
    rho = dim_v - 1
    for i in range(dim_v):
        res = mu[i] - (torch.sum(mu[:i]) - z) / (i + 1)
        if res <= 0:
            rho = i - 1
            break

    # rho = min(rho, dim_v - rho_tolerance)
    # even if l1 norm is satisfied, kill some weights
    if rho > dim_v - rho_tolerance:
        rho = dim_v - rho_tolerance
        # theta = mu[rho] # subtract mu rho from everything
        theta = torch.zeros_like(mu) 
        theta[rho:] = mu[rho]
        # should just kill the last "rho tolerance" weights, keeping all before
        print("artificially killing some weights, smallest is ", mu[rho])
    else:
        # theta = mu[dim_v - rho_tolerance :].mean()
        theta = (torch.sum(mu[:rho]) - z) / (rho + 1)

    trimmed = (mu - theta).clamp(min=0)

    # theta = (torch.sum(mu[:rho]) - z) / (rho + 1)
    # trimmed = (mu - theta).clamp(min=0)

    print("rho", rho)
    # print("rho, theta", rho, theta)
    model.fc.weight.data = (trimmed[p] * signs).reshape(model.fc.weight.shape)
    print("num zeros", (model.fc.weight == 0).sum().item())


def proj(model, z, device="cuda"):
    ## minimizing ||v||_1, so projecting last fc layer to l1 ball
    v = model.fc.weight.data.flatten()
    v = torch.cat((v, torch.zeros(1).to(device)))
    n = v.shape[0]

    U = torch.arange(n).to(device)
    s = 0
    rho = 0

    while len(U) > 0:
        k = torch.randint(len(U), (1,))[0]
        G = U[torch.where(v[U] >= v[U[k]])]
        L = U[torch.where(v[U] < v[U[k]])]

        del_rho = len(G)
        del_s = torch.sum(v[G])

        if (s + del_s) - (rho + del_rho) * v[U[k]] < z:
            s += del_s
            rho += del_rho
            U = L
        else:
            U = torch.cat((G[:k], G[k + 1 :]))

    if rho == n - 1:
        return

    theta = (s - z) / rho
    result = (v - theta).clamp(min=0)
    result = result[:-1]
    model.fc.weight.data = result.reshape(model.fc.weight.shape)
    return

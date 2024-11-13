from timm.optim import create_optimizer, create_optimizer_v2

def create_optimizer(args, model):
    """
    Create an optimizer based on the given arguments and parameters.

    Args:
        args (argparse.Namespace): The arguments containing optimizer configuration.
        model : The model to optimize.

    Returns:
        torch.optim.Optimizer: The created optimizer.

    Raises:
        AssertionError: If an invalid optimizer is specified.
    """
    
    if hasattr(model, 'get_parameters'):
        opt_params = model.get_parameters(
            weight_decay=args.weight_decay,
        )
        parameters = opt_params['parameters']
        args.weight_decay = opt_params.get('weight_decay', args.weight_decay)
    else:
        parameters = model.parameters()
    

    opt_args = {
        'opt': args.opt, 
        'lr': args.lr, 
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
    }
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None and args.opt != 'sgd':
        opt_args['betas'] = args.opt_betas
    if 'lamb' in args.opt:
        clip_grad = args.clip_grad if args.clip_grad > 0 else 1.0
        opt_args['max_grad_norm'] = clip_grad
        
    optimizer = create_optimizer_v2(parameters, **opt_args)

    
        
    print(f"Using optimizer: {optimizer.__class__.__name__}")
    print(f"Optimizer arguments: {opt_args}")

    return optimizer
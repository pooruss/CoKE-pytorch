# model config
def init_coke_net_config(args, logger, print_config = True):
    config = dict()
    config["max_seq_len"] = args.max_seq_len
    config["hidden_size"] = args.hidden_size
    config["num_hidden_layers"] = args.num_hidden_layers
    config["num_attention_heads"] = args.num_attention_heads
    config["vocab_size"] = args.vocab_size
    config["num_relations"] = args.num_relations
    config["weight_sharing"] = args.weight_sharing
    config["max_position_embeddings"] = args.max_position_embeddings
    config["hidden_act"] = args.hidden_act
    config["dropout"] = args.dropout
    config["hidden_dropout"] = args.hidden_dropout
    config["attention_dropout"] = args.attention_dropout
    config["initializer_range"] = args.initializer_range
    config["intermediate_size"] = args.intermediate_size
    config["activation_dropout"] = args.activation_dropout
    config["soft_label"] = args.soft_label
    # config["gpus"] = args.gpus
    config["use_cuda"] = args.use_cuda
    if print_config is True:
        logger.info('----------- CoKE Network Configuration -------------')
        for arg, value in config.items():
            logger.info('%s: %s' % (arg, value))
        logger.info('------------------------------------------------')
    return config

# train config
def init_train_config(args, logger, print_config = True):
    config = dict()
    config["do_train"] = args.do_train
    config["do_val"] = args.do_val
    config["do_test"] = args.do_test
    config["batch_size"] = args.batch_size
    config["epoch"] = args.epoch
    config["learning_rate"] = args.learning_rate
    config["use_cuda"] = args.use_cuda
    config["model_name"] = args.model_name
    config["gpus"] = args.gpus
    config["nodes"] = args.nodes
    config["skip_steps"] = args.skip_steps
    config["save_path"] = args.save_path
    config["checkpoint_num"] = args.checkpoint_num
    if print_config is True:
        logger.info('----------- Train Configuration -------------')
        for arg, value in config.items():
            logger.info('%s: %s' % (arg, value))
        logger.info('------------------------------------------------')
    return config
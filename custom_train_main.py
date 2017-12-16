# python OpenNMT-py/train.py -data data/ted-nl-nl/data/ted_nl_nl -save_model autoencoder_CPU -train_from data/Ted-en-nl/model-en-nl.pt
def custom_main():
    # Load train and validate data.
    print("Loading train and validate data from '%s'" % opt.data)
    train = torch.load(opt.data + '.train.pt')
    valid = torch.load(opt.data + '.valid.pt')
    print(' * number of training sentences: %d' % len(train))
    print(' * maximum batch size: %d' % opt.batch_size)

    ## ----- Load pre-trained freezed decoder -----
    # Load checkpoint
    print('Loading decoder from %s' % opt.train_from)
    checkpoint = torch.load(opt.train_from,
                            map_location=lambda storage, loc: storage)
    model_opt = checkpoint['opt']
    model_opt.save_model = opt.save_model

    # Load fields from pre-trained model 1 using checkpoint
    fields_checkpoint = load_fields(train, valid, checkpoint)

    # Build pre_trained model.
    decoder_pre = build_model(model_opt, opt, fields_checkpoint, checkpoint).decoder

    # Freeze decoder weights during training time
    print('Freezing model decoder weights...\n')
    for param in decoder_pre.parameters():
        param.requires_grad = False

    ## ----- Create autoencoder and attach pre-trained freezed decoder -----
    opt.train_from = ''
    # Load fields without using checkpoint for the untrained autencoder
    fields = load_fields(train, valid, None)

    # Collect features.
    src_features = collect_features(train, fields)
    for j, feat in enumerate(src_features):
        print(' * src feature %d size = %d' % (j, len(fields[feat].vocab)))

    # Build new model
    model = build_model(opt, opt, fields, None)

    # Use pre-trained decoder for new model and freeze weights
    print('Replacing untrained decoder with the pre-trained freezed decoder...')
    model.decoder = decoder_pre
    tally_parameters(model)
    check_save_model_path()

    # Build optimizer.
    optim = build_optim(model, None)

    # Do training.
    train_model(model, train, valid, fields, optim)

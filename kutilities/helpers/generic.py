def get_model_desc(model):
    """
    Generates a small description of a Keras model. Suitable for generating footer descriptions for charts.
    :param model:
    :return:
    """
    desc = []

    conf = model.get_config()

    for layer in (conf["layers"] if "layers" in conf else conf):
        if "layer" in layer["config"]:
            name = "_".join([layer['class_name'], layer["config"]['layer']['class_name']])
            config = layer["config"]['layer']["config"]

        else:
            name = layer['class_name']
            config = layer["config"]
        params = []
        try:
            params.append(config["p"])
        except:
            pass
        try:
            params.append(config["sigma"])
        except:
            pass
        try:
            params.append(config["output_dim"])
        except:
            pass
        try:
            params.append(config["activation"])
        except:
            pass
        try:
            params.append(config['l2'])
        except:
            pass

        desc.append(name + "({})".format(",".join([str(p) for p in params])))

    description = " -> ".join(desc)
    try:
        description += " : [optimizer= {}, clipnorm={} - batch_size={}]".format(model.optimizer.__class__.__name__,
                                                                                model.optimizer.clipnorm,
                                                                                model.model.history.params[
                                                                                    'batch_size'])
    except:
        pass
    return description

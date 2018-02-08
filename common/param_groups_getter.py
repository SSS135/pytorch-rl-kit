def get_param_groups(model):
    bias_group = []
    others_group = []
    for n, p in model.named_parameters():
        if n.find('bias') != -1 or n.find('beta') != -1:
            bias_group.append(p)
        else:
            others_group.append(p)
    return [dict(params=others_group),
            dict(params=bias_group, weight_decay=0)]
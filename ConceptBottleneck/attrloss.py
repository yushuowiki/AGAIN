import torch


def logic_loss(outputs, aux_outputs, attr_labels_var):
    """
    Args:
        outputs[112,16]
        aux_outputs: [112,16]
        attr_labels_var[16,112]
    """

    # outputs_tensor = torch.stack(outputs).cuda() if torch.cuda.is_available() else torch.stack(outputs)
    # aux_outputs_tensor = torch.stack(aux_outputs).cuda() if torch.cuda.is_available() else torch.stack(aux_outputs)
    crm_output = torch.zeros(outputs.shape[0], outputs.shape[0]).cuda() if torch.cuda.is_available() else torch.zeros(outputs.shape[0], outputs.shape[0])
    crm_aux_outputs = torch.zeros(aux_outputs.shape[0], aux_outputs.shape[0]).cuda() if torch.cuda.is_available() else torch.zeros(aux_outputs.shape[0], aux_outputs.shape[0])

    # outputs_tensor = change_bool(outputs_tensor)
    outputs_tensor = torch.where(outputs > 0.5, 1.0, 0.0)
    aux_outputs_tensor = torch.where(aux_outputs > 0.5, 1.0, 0.0)
    # aux_outputs_tensor = change_bool(aux_outputs_tensor)
    for i in range(outputs_tensor.shape[1]):
        crm_output += outputs_tensor[:, i] @ outputs_tensor[:, i].T
        crm_aux_outputs += aux_outputs_tensor[:, i] @ aux_outputs_tensor[:, i].T

    crm_output = crm_output / outputs_tensor.shape[1]
    crm_aux_outputs = crm_aux_outputs / aux_outputs_tensor.shape[1]

    class_attr_labels = attr_labels_var[0, :]
    class_attr_labels = torch.unsqueeze(class_attr_labels, 0)
    crm = class_attr_labels.T @ class_attr_labels
    #print(crm)
    #print(crm_output)
    #print(crm_aux_outputs)

    MSE = torch.nn.MSELoss(reduce=True, size_average=True)
    loss1 = MSE(crm_output,crm)
    loss2 = MSE(crm_aux_outputs, crm)
    print("logic loss: "+str((1.0 * loss1) + (0.4 * loss2)))
    return (1.0 * loss1) + (0.4 * loss2)

def change_bool(tensor):
    torch.where(tensor > 0.5, 1, 0)
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            if tensor[i, j] > 0.5:
                tensor[i, j] = 1
            else:
                tensor[i, j] = 0
    return tensor
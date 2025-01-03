
def training_losses(model, x_start, t,loss_type, index,condition, c= 0.0, device = None, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: input tensor.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if device is None:
          device = next(model.parameters()).device

        x_start = x_start.to(device)
        t = t.to(device)
        condition = condition.to(device) if condition is not None else None

        if noise is None:
          noise = th.randn_like(x_start, device=device)
        else:
          noise = noise.to(device)
        x_t = q_sample(x_start, t, noise=noise)

        terms = {}
        model_output = model(x_t, t, context=condition,**model_kwargs)
        model_output = x_t-model_output


        if loss_type == "MSE":


            terms["guide"] = mean_flat((model_output-x_start) ** 2, batch)
            terms["iter"] = mean_flat((model_output-condition) ** 2, batch)


            #model.module.update_xbar(model_output,index)

        elif loss_type == "ph_loss":

            terms["guide"]= ph_loss(model_output,x_start,c)
            terms["iter"]= ph_loss(model_output,condition,c)


        else:
            raise NotImplementedError()

        return terms, model_output

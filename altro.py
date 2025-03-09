def FGSM_u(correct_examples, correct_labels, attack_index, eps):
    x_att = correct_examples[attack_index]
    x_att.requires_grad_()

    # True label of x_att
    y_att = correct_labels[attack_index]

    prova.network.eval()

    # log p(y_att | x_att)
    log_p_y_att = prova.network(x_att)[0][y_att]  # Directly use log-softmax
    log_p_y_att.backward()

    # Generate adversarial example
    x_adv = x_att - torch.sign(x_att.grad) * eps

    # project to image space
    x_adv = torch.clamp(x_adv, -0.424, 2.821)

    # Vector of log p(y | x_adv)
    log_p_y = prova.network(x_adv)

    # Predicted class and probability for x_adv
    y_adv = log_p_y.argmax()
    log_p_y_adv = log_p_y[0][y_adv]

    # Convert log probabilities to probabilities
    p_y_att = torch.exp(log_p_y_att).item()
    p_y_adv = torch.exp(log_p_y_adv).item()

    show_result(x_att, x_adv, y_att, y_adv, p_y_att, p_y_adv, "FGSM")

def FGSM_t(x_att, y_tar, eps, verbose):
    # controlla che y_tar sia diversoda y_att

    # x_att = correct_examples[attack_index]
    x_att.requires_grad_()

    prova.network.eval()
    f = prova.TargetedObjective(prova.network, y_tar)

    # log_p_y_tar_given_x_att = f(x_att)
    y = f(x_att)

    '''with torch.no_grad():
        print("f_x_att: ", log_p_y_tar_given_x_att.item())
        print("p(y_tar|x_att): ", torch.exp(-log_p_y_tar_given_x_att).item())
        print()ì'''

    # log_p_y_tar = prova.network(x_att)[0][y_tar]

    y.backward()

    x_adv = x_att - torch.sign(x_att.grad) * eps

    return x_adv

def BIM_u(correct_examples, correct_labels, attack_index, eps, T):
    prova.network.eval()

    x_att = correct_examples[attack_index]

    # True label of x_att
    y_att = correct_labels[attack_index]

    x_i = x_att

    alpha = eps / T
    for i in range(T):
        x_i.requires_grad_()

        prova.network.zero_grad()

        # log p(y_att | x_i) (scalar)
        log_p_y_att = prova.network(x_i)[0][y_att]
        log_p_y_att.backward()
        x_i = x_i - alpha * torch.sign(x_i.grad)

        # in teoria non serve se il passo è eps/T. ma se lo metti qui funziona per tutti gli alpha
        # x_i = torch.clamp(x_i, -0.424, 2.821)

        # perche altrimenti non diventa piu leaf
        x_i = x_i.detach()

    x_adv = x_i

    # serve qui perche eps-ball potrebbe uscire da [0,1]^(32x32)
    # se togli il NN l'attacco tende a vincere piu spesso, purtroppo l'immagine non è valida
    x_adv = torch.clamp(x_adv, -0.424, 2.821)

    with torch.no_grad():
        # log p(y_att | x_att) (scalar)
        log_p_y_att = prova.network(x_att)[0][y_att]

        # Vector of log p(y | x_adv)
        log_p_y = prova.network(x_adv)

    # classe predetta per x_adv
    y_adv = log_p_y.argmax()
    # scalar log p(y_adv | x_adv)
    log_p_y_adv = log_p_y[0][y_adv]

    p_y_att = torch.exp(log_p_y_att).item()
    p_y_adv = torch.exp(log_p_y_adv).item()

    show_result(x_att, x_adv, y_att, y_adv, p_y_att, p_y_adv, "I-FGSM")

def PGD_u(correct_examples, correct_labels, attack_index, eps, T, alpha):
    prova.network.eval()

    x_att = correct_examples[attack_index]

    # True label of x_att
    y_att = correct_labels[attack_index]

    # x_0 = x_att + U(-eps,eps)
    x_i = x_att + (torch.rand_like(x_att) * 2 * eps - eps)

    for i in range(T):
        # proietto su eps-ball, metto all'inizio perche x_0 potrebbe gia essere fuori
        x_i = torch.clamp(x_i, x_att - eps, x_att + eps)

        x_i.requires_grad_()

        prova.network.zero_grad()

        # log p(y_att | x_i) (scalar)
        log_p_y_att = prova.network(x_i)[0][y_att]
        log_p_y_att.backward()
        x_i = x_i - alpha * torch.sign(x_i.grad)

        # perche altrimenti non diventa piu leaf
        x_i = x_i.detach()

    x_adv = x_i

    # proietto su image space [0,1]^(32x32)
    x_adv = torch.clamp(x_adv, -0.424, 2.821)

    with torch.no_grad():
        # log p(y_att | x_att) (scalar)
        log_p_y_att = prova.network(x_att)[0][y_att]

        # Vector of log p(y | x_adv)
        log_p_y = prova.network(x_adv)

    # classe predetta per x_adv
    y_adv = log_p_y.argmax()
    # scalar log p(y_adv | x_adv)
    log_p_y_adv = log_p_y[0][y_adv]

    p_y_att = torch.exp(log_p_y_att).item()
    p_y_adv = torch.exp(log_p_y_adv).item()

    show_result(x_att, x_adv, y_att, y_adv, p_y_att, p_y_adv, "PROJ GRAD")

def CW_t(correct_examples, correct_labels, attack_index, y_tar, alpha, lmbda, T, ord):
    prova.network.eval()
    x_att = correct_examples[attack_index]

    print(x_att.shape)

    y_att = correct_labels[attack_index]
    x_k = x_att

    previous_grad_phi = None

    for k in range(T):
        ## togli
        with torch.no_grad():
            log_p_y_tar = prova.network(x_k)[0, y_tar]
            print("f: ", -log_p_y_tar)
            print("p: ", torch.exp(log_p_y_tar))

        x_k.requires_grad_()

        # vettore
        log_p_y = prova.network(x_k)[0]

        with torch.no_grad():
            # trovo i*
            masked_log_p_y = log_p_y.clone()
            masked_log_p_y[y_tar] = float('-inf')
            i_star = torch.argmax(masked_log_p_y).item()

        # grad g_i*
        log_p_y[i_star].backward(retain_graph=True)
        grad_g_i_star = x_k.grad.clone()

        prova.network.zero_grad()
        x_k.grad.zero_()

        # grad_g_tar
        log_p_y[y_tar].backward(retain_graph=True)
        grad_g_tar = x_k.grad.clone()

        # calcolo il grad con pytorch
        prova.network.zero_grad()
        x_k.grad.zero_()
        x_att.grad.zero_()
        # 2-norm
        # phi = torch.square(torch.norm( x_k - x_att)) + lmbda*torch.relu(log_p_y[i_star] - log_p_y[y_tar]  )
        # inf norm
        phi = torch.square(torch.linalg.vector_norm(x_k - x_att, ord=float(ord))) + lmbda * torch.relu(
            log_p_y[i_star] - log_p_y[y_tar])
        phi.backward()
        grad_phi2 = x_k.grad

        # analiticamente
        # grad_phi = 2*( x_k-x_att) + lmbda*(1 if log_p_y[i_star] - log_p_y[y_tar] > 0 else 0)*(grad_g_i_star - grad_g_tar)

        # print(f"Maximum difference: { torch.max(torch.abs(phi_grad2 - grad_phi)).item()}")
        # print(f"||grad||: {torch.linalg.vector_norm(grad_phi2)}")
        # if previous_grad_phi is not None:
        # print(f"||grad_xk - grad_xk-1||: {torch.linalg.vector_norm(grad_phi2-previous_grad_phi)}")
        previous_grad_phi = grad_phi2.clone()

        x_k = x_k - alpha * grad_phi2

        # ma in teoria devi proiettare solo alla fine non ogni volta??
        # in teoria non va bene se usi norma2
        x_k = torch.clamp(x_k, -0.424, 2.821)

        x_k.detach_()

    x_adv = x_k

    with torch.no_grad():
        # log p(y_att | x_att) (scalar)
        log_p_y_att = prova.network(x_att)[0][y_att]

        # Vector of log p(y | x_adv)
        log_p_y = prova.network(x_adv)

    # classe predetta per x_adv
    y_adv = log_p_y.argmax()
    # scalar log p(y_adv | x_adv)
    log_p_y_adv = log_p_y[0][y_adv]

    p_y_att = torch.exp(log_p_y_att).item()
    p_y_adv = torch.exp(log_p_y_adv).item()

    show_result(x_att, x_adv, y_att, y_adv, p_y_att, p_y_adv, "CW")

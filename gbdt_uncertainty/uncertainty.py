import numpy as np


def kl_divergence_class(probs1, probs2, epsilon=1e-10):
    return np.sum(probs1 * (np.log(probs1 + epsilon) - np.log(probs2 + epsilon)), axis=1)

def expected_pairwise_kl_divergence_class(probs, epsilon=1e-10):
    kl = 0.0
    for i in range(probs.shape[1]):
        for j in range(probs.shape[1]):
            kl += kl_divergence_class(probs[:, i, :], probs[:, j, :], epsilon)
    return kl

def entropy(probs, epsilon=1e-10):
    log_probs = -np.log(probs + epsilon)
    return np.sum(probs * log_probs, axis=1)

def entropy_of_expected_class(probs, epsilon=1e-10):
    mean_probs = np.mean(probs, axis=0)
    log_probs = -np.log(mean_probs + epsilon)
    return np.sum(mean_probs * log_probs, axis=1)


def expected_entropy_class(probs, epsilon=1e-10):
    log_probs = -np.log(probs + epsilon)

    return np.mean(np.sum(probs * log_probs, axis=2), axis=0)

def mutual_information_class(probs, epsilon):
    eoe = entropy_of_expected_class(probs, epsilon)
    exe = expected_entropy_class(probs, epsilon)
    return eoe - exe


def ensemble_uncertainties_classification(probs, epsilon=1e-10):
    """

    :param probs: Tensor [num_models, num_examples, num_classes]
    :return: Dictionary of uncertaintties
    """
    mean_probs = np.mean(probs, axis=0)
    mean_lprobs = np.mean(np.log(probs + epsilon), axis=0)
    conf = np.max(mean_probs, axis=1)

    eoe = entropy_of_expected_class(probs, epsilon)
    exe = expected_entropy_class(probs, epsilon)
    mutual_info = eoe - exe

    epkl = np.sum(-mean_probs * mean_lprobs, axis=1) - exe

    uncertainty = {'confidence': conf,
                   'entropy_of_expected': eoe,
                   'expected_entropy': exe,
                   'mutual_information': mutual_info,
                   'epkl': epkl,
                   'rmi': epkl - mutual_info
                   }

    return uncertainty


def normal_KL(params1, params2, epsilon=1e-20):
    mu_1 = params1[0]
    mu_2 = params2[0]

    logvar1 = np.log(params1[1] + epsilon)
    logvar2 = np.log(params2[1] + epsilon)

    mean_term = 0.5 * np.exp(2 * np.log(np.abs(mu_1 - mu_2)) - logvar2)
    sigma_term = 0.5 * (np.exp(logvar1 - logvar2) - 1.0 + logvar2 - logvar1)

    return mean_term + sigma_term


def epkl_reg(preds):
    """
    preds: array [n_samples, n_models, 2]
    """
    M = preds.shape[1]
    EPKL = []
    for pred in preds:
        epkl = 0.0
        for i, pr1 in enumerate(pred):
            for j, pr2 in enumerate(pred):
                if i != j:
                    epkl += normal_KL(pr1, pr2)

        epkl = epkl / (M * (M - 1))
        EPKL.append(epkl)
    return np.asarray(EPKL)


def ensemble_uncertainties_regression(preds):
    """
    preds: array [n_samples, n_models, 2] - last dim ins mean, var
    """
    epkl = epkl_reg(preds)

    var_mean = np.var(preds[:, :, 0], axis=1)
    mean_var = np.mean(preds[:, :, 1], axis=1)

    uncertainty = {'tvar': var_mean + mean_var,
                   'mvar': mean_var,
                   'varm': var_mean,
                   'epkl': epkl}

    return uncertainty

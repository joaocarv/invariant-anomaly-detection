import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Make the code better
def huber_loss(y_true, y_pred):
    return F.huber_loss(y_true, y_pred)


def adjust_binary_cross_entropy(y_true, y_pred):
    return F.binary_cross_entropy(y_true, torch.pow(y_pred, 2))


class MMDLoss(nn.Module):

    def __init__(self, pooled=False):
        super().__init__()
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.pooled = pooled

    def forward(self, env_label, latent, sigmas=[0.5, 1.5, 5.0], use_cosine=False, rich_feature=False,
                no_kernel=False, sum_of_distances=False, normalize_per_channel=True):

        if env_label.eq(env_label[0]).sum() == len(env_label):
            return torch.tensor(0.0)
        cost = []
        envs = env_label.unique(sorted=True)
        if no_kernel:
            loss_func = self._mmd_two_distribution_no_kernel
        elif use_cosine:
            loss_func = self._mmd_cosine_two_distribution
        elif sum_of_distances:
            loss_func = self._sum_of_distances
        else:
            loss_func = self._mmd_two_distribution
        for i in envs:
            for j in envs:
                if i >= j:
                    continue
                domain_i = torch.where(torch.eq(env_label, i))[0]
                domain_j = torch.where(torch.eq(env_label, j))[0]
                if len(domain_i) < 1 or len(domain_j) < 1:
                    continue
                if rich_feature:
                    _, channel_size, width, height = latent.shape
                    single_res = 0.0
                    for k in range(channel_size):
                        single_res += loss_func(latent[domain_i, k:k + 1],
                                                latent[domain_j, k:k + 1],
                                                sigmas=sigmas)
                    single_res /= (channel_size * width * height)
                else:
                    if not normalize_per_channel:
                        latent = latent.reshape(latent.shape[0], -1)
                    single_res = loss_func(latent[domain_i],
                                           latent[domain_j],
                                           sigmas=sigmas)
                cost.append(single_res)

        cost = torch.cat(cost)
        return torch.sum(cost)

    def _mmd_cosine_two_distribution(self, source, target, sigmas):
        sigmas = torch.tensor(sigmas).to(self.device)
        source = F.normalize(source, dim=1)
        target = F.normalize(target, dim=1)
        if not self.pooled:
            source = source.reshape(source.shape[0], -1)
            target = target.reshape(target.shape[0], -1)
        xy = self._cosine_kernel(source, target, sigmas)
        xx = self._cosine_kernel(source, source, sigmas)
        yy = self._cosine_kernel(target, target, sigmas)
        return torch.abs(xx + yy - 2 * xy)

    def _cosine_kernel(self, x, y, sigmas):
        beta = 1. / (2. * (torch.unsqueeze(sigmas, 1)))
        dist = self._compute_pairwise_cosine_similarity(x, y)
        dot = torch.matmul(beta, torch.reshape(dist, (1, -1)))
        exp = torch.exp(dot)
        return torch.mean(exp, 1)

    def _compute_pairwise_cosine_similarity(self, x, y):
        matrix = torch.mm(x, y.T)
        return matrix

    def _sum_of_distances(self, source, target, sigmas):
        source = F.normalize(source, dim=1)
        target = F.normalize(target, dim=1)
        if not self.pooled:
            channel, width, height = source.shape[1:]
            source = source.reshape(source.shape[0], -1)
            target = target.reshape(target.shape[0], -1)
        else:
            channel, width, height = (1, 1, 1)
        mean_dist = 0.5 * self._compute_pairwise_distances(source, target).mean()
        return (mean_dist / channel).reshape((1, 1))

    def _mmd_two_distribution_no_kernel(self, source, target, sigmas):
        source = F.normalize(source, dim=1)
        target = F.normalize(target, dim=1)
        if not self.pooled:
            channel, width, height = source.shape[1:]
            source = source.reshape(source.shape[0], -1)
            target = target.reshape(target.shape[0], -1)
        else:
            channel, width, height = (1, 1, 1)
        xx = self._compute_pairwise_distances(source, source).mean()
        yy = self._compute_pairwise_distances(target, target).mean()
        xy = self._compute_pairwise_distances(source, target).mean()
        return (0.5 / (channel * width * height)) * torch.abs((xx + yy - 2 * xy)).reshape((1, 1))

    def _mmd_two_distribution(self, source, target, sigmas):
        """
        compute mmd loss between two distributions
        :param source: [num_samples, num_features]
        :param target: [num_samples, num_features]
        :return:
        """

        sigmas = torch.tensor(sigmas).to(self.device)
        source = F.normalize(source, dim=1)
        target = F.normalize(target, dim=1)
        if not self.pooled:
            channel, width, height = source.shape[1:]
            source = source.reshape(source.shape[0], -1)
            target = target.reshape(target.shape[0], -1)
        else:
            channel, width, height = (1, 1, 1)
        xy = self._rbf_kernel(source, target, sigmas)
        xx = self._rbf_kernel(source, source, sigmas)
        yy = self._rbf_kernel(target, target, sigmas)
        return (0.5 / (channel * width * height)) * torch.abs(xx + yy - 2 * xy)

    def _rbf_kernel(self, x, y, sigmas):
        """
        compute the rbf kernel value
        :param x: [num_x_samples, num_features]
        :param y: [num_y_samples, num_features]
        :param sigmas: sigmas need to use
        :return: single value of x, y kernel
        """
        beta = 1. / (2. * (torch.unsqueeze(sigmas, 1)))
        dist = self._compute_pairwise_distances(x, y)
        dot = -torch.matmul(beta, torch.reshape(dist, (1, -1)))
        exp = torch.exp(dot)
        return torch.mean(exp, 1)

    def _compute_pairwise_distances(self, x, y):
        """Computes the squared pairwise Euclidean distances between x and y.
        Args:
          x: a tensor of shape [num_x_samples, num_features]
          y: a tensor of shape [num_y_samples, num_features]
        Returns:
          a distance matrix of dimensions [num_x_samples, num_y_samples].
        """
        return torch.sum((torch.unsqueeze(x, 2) - torch.transpose(y, 0, 1)) ** 2, dim=1)

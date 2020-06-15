import torch
import torch.nn as nn
import numpy as np



class DMSpline(nn.Module):
    """Attempt at re-creating spline regression in contextual embedding context

    Adaped from Distributed Memory version of Paragraph Vectors.

    Parameters
    ----------
    vec_dim: int
        Dimensionality of vectors to be learned (for paragraphs and words).

    num_splines: int
        Number of splines

    num_words: int
        Number of distinct words in a daset (i.e. vocabulary size).
    """
    def __init__(self, vec_dim, num_splines, num_words):
        super(DMSpline, self).__init__()
        # matrix of spline coefficients
        self._D = nn.Parameter(
            torch.randn(num_splines, num_words, vec_dim), requires_grad=True)
        # word matrix
        self._W = nn.Parameter(
            torch.randn(num_words, vec_dim), requires_grad=True)
        # output layer parameters
        self._O = nn.Parameter(
            torch.FloatTensor(vec_dim, num_words).zero_(), requires_grad=True)

        self.vec_dim = vec_dim

    def forward(self, x_vals, knot_ids, context_ids, target_noise_ids):
        """Sparse computation of scores (unnormalized log probabilities)
        that should be passed to the negative sampling loss.

        Parameters
        ----------

        x_vals: torch.Tensor of size (batch_size,)
            Values for attributes to train against

        knot_ids: torch.Tensor of size (batch_size,)
            The knot interval corresponding to each element of attr_vals

        context_ids: torch.Tensor of size (batch_size, num_context_words)
            Vocabulary indices of context words.

        target_noise_ids: torch.Tensor of size (batch_size, num_noise_words + 1)
            Vocabulary indices of target and noise words. The first element in
            each row is the ground truth index (i.e. the target), other
            elements are indices of samples from the noise distribution.

        Returns
        -------
            autograd.Variable of size (batch_size, num_noise_words + 1)
        """
        # combine a paragraph vector with word vectors
        # input (context) words

        context_ids_t = context_ids.permute(1,0)
        x_vals_rep = x_vals.expand(self.vec_dim, context_ids_t.shape[0], knot_ids.shape[0]).permute(2, 1, 0)
        beta_times_x = torch.mul(self._D[knot_ids, context_ids_t, :].permute(1, 0, 2), x_vals_rep)

        x = torch.mean(torch.add(beta_times_x, self._W[context_ids, :]), dim=1)

        # print('Shape of x: ', x.shape)
        # sparse computation of scores (unnormalized log probabilities)
        # for negative sampling
        return (torch.bmm(
            x.unsqueeze(1),
            self._O[:, target_noise_ids].permute(1, 0, 2)).squeeze(), self._D)


    def get_paragraph_vector(self, index):
        return torch.mean(self._D[index, :, :], dim=1).data.tolist()


    def get_baseline_word_vector(self, index):
        return self._W[index, :].data.tolist()


    def get_paragraph_word_vector(self, doc_index, word_index, x_val):
        to_return = self._D[doc_index, word_index, :].data.tolist() * np.repeat(x_val, self.vec_dim)
        return to_return


class DM(nn.Module):
    """Distributed Memory version of Paragraph Vectors.

    Parameters
    ----------
    vec_dim: int
        Dimensionality of vectors to be learned (for paragraphs and words).

    num_docs: int
        Number of documents in a dataset.

    num_words: int
        Number of distinct words in a daset (i.e. vocabulary size).
    """
    def __init__(self, vec_dim, num_docs, num_words):
        super(DM, self).__init__()
        # paragraph matrix
        self._D = nn.Parameter(
            torch.randn(num_docs, num_words, vec_dim), requires_grad=True)
        # word matrix
        self._W = nn.Parameter(
            torch.randn(num_words, vec_dim), requires_grad=True)
        # output layer parameters
        self._O = nn.Parameter(
            torch.randn(vec_dim, num_words), requires_grad=True)
    

    def forward(self, context_ids, doc_ids, target_noise_ids):
        """Sparse computation of scores (unnormalized log probabilities)
        that should be passed to the negative sampling loss.

        Parameters
        ----------
        context_ids: torch.Tensor of size (batch_size, num_context_words)
            Vocabulary indices of context words.

        doc_ids: torch.Tensor of size (batch_size,)
            Document indices of paragraphs.

        target_noise_ids: torch.Tensor of size (batch_size, num_noise_words + 1)
            Vocabulary indices of target and noise words. The first element in
            each row is the ground truth index (i.e. the target), other
            elements are indices of samples from the noise distribution.

        Returns
        -------
            autograd.Variable of size (batch_size, num_noise_words + 1)
        """
        # combine a paragraph vector with word vectors of
        # input (context) words


        # batch_docs = self._D[doc_ids, :, :] # batch_size x vocab_size x dim
        
        # bsz, _, dim = batch_docs.size()
        # n_context = context_ids.size(1)

        # doc_context_words = torch.FloatTensor(bsz, n_context, dim)
        # avg_emb_context_words = torch.FloatTensor(bsz, n_context, dim)

        # for i in range(0, bsz):
        #     doc_context_words[i,:,:] = batch_docs[i, context_ids[i], :]  # item is [n_context, dim]
        #     avg_emb_context_words[i, :, :] = self._W[context_ids[i], :] # item is [n_context, dim]

        # x = torch.sum(
        #         torch.add(doc_context_words, avg_emb_context_words), dim=1
        #     ).unsqueeze(1) # batch_size x 1 x vec_dim


        # num_noise_words = target_noise_ids.size(1)
        # curr_target_noise_words = torch.FloatTensor(bsz, dim, num_noise_words)
        # for i in range(0, bsz):
        #     curr_target_noise_words[i, :, :] = self._O[:, target_noise_ids[i]]

        # result = torch.bmm(x, curr_target_noise_words)
        # result = result.squeeze() # batch_size x num_noise_words

        # return result
        

        context_ids_t = context_ids.transpose(0,1) # context_size x batch_size

        # x = torch.mean(
        #         torch.add(self._D[doc_ids, context_ids_t, :].transpose(0,1), self._W[context_ids, :]), dim=1
        #     ) # batch_size x vec_dim

        x = torch.sum(
                torch.add(self._D[doc_ids, context_ids_t, :].transpose(0,1), self._W[context_ids, :]), dim=1
            ) # batch_size x vec_dim

        # sparse computation of scores (unnormalized log probabilities)
        # for negative sampling
        result = torch.bmm(x.unsqueeze(1), self._O[:, target_noise_ids].permute(1, 0, 2))
        result = result.squeeze()

        return result


    def get_paragraph_vector(self, index):
        return torch.mean(self._D[index, :, :], dim=1).data.tolist()


    def get_baseline_word_vector(self, index):
        return self._W[index, :].data.tolist()


    def get_paragraph_word_vector(self, doc_index, word_index):
        return self._D[doc_index, word_index, :].data.tolist()


class DBOW(nn.Module):
    """Distributed Bag of Words version of Paragraph Vectors.

    Parameters
    ----------
    vec_dim: int
        Dimensionality of vectors to be learned (for paragraphs and words).

    num_docs: int
        Number of documents in a dataset.

    num_words: int
        Number of distinct words in a daset (i.e. vocabulary size).
    """
    def __init__(self, vec_dim, num_docs, num_words):
        super(DBOW, self).__init__()
        # paragraph matrix
        self._D = nn.Parameter(
            torch.randn(num_docs, vec_dim), requires_grad=True)
        # output layer parameters
        self._O = nn.Parameter(
            torch.FloatTensor(vec_dim, num_words).zero_(), requires_grad=True)

    def forward(self, doc_ids, target_noise_ids):
        """Sparse computation of scores (unnormalized log probabilities)
        that should be passed to the negative sampling loss.

        Parameters
        ----------
        doc_ids: torch.Tensor of size (batch_size,)
            Document indices of paragraphs.

        target_noise_ids: torch.Tensor of size (batch_size, num_noise_words + 1)
            Vocabulary indices of target and noise words. The first element in
            each row is the ground truth index (i.e. the target), other
            elements are indices of samples from the noise distribution.

        Returns
        -------
            autograd.Variable of size (batch_size, num_noise_words + 1)
        """
        # sparse computation of scores (unnormalized log probabilities)
        # for negative sampling
        return torch.bmm(
            self._D[doc_ids, :].unsqueeze(1),
            self._O[:, target_noise_ids].permute(1, 0, 2)).squeeze()

    def get_paragraph_vector(self, index):
        return self._D[index, :].data.tolist()

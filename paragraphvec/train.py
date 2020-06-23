import time
from sys import float_info, stdout

import fire
import torch
from torch.optim import Adam, SGD

from data import load_dataset, NCEData
from dataset import load_and_cache_data, make_dataloader
from loss import NegativeSampling, NegativeSamplingWithSpline
from models import DM, DBOW, DMSpline
from utils import save_training_state


def start(data_file_name,
          num_noise_words,
          vec_dim,
          num_epochs,
          batch_size,
          lr,
          model_ver='dm',
          context_size=0,
          vec_combine_method='sum',
          save_all=True,
          generate_plot=True,
          max_generated_batches=5,
          num_workers=1):
    """Trains a new model. The latest checkpoint and the best performing
    model are saved in the *models* directory.

    Parameters
    ----------
    data_file_name: str
        Name of a file in the *data* directory.

    model_ver: str, one of ('dm', 'dbow'), default='dbow'
        Version of the model as proposed by Q. V. Le et al., Distributed
        Representations of Sentences and Documents. 'dbow' stands for
        Distributed Bag Of Words, 'dm' stands for Distributed Memory.

    vec_combine_method: str, one of ('sum', 'concat'), default='sum'
        Method for combining paragraph and word vectors when model_ver='dm'.
        Currently only the 'sum' operation is implemented.

    context_size: int, default=0
        Half the size of a neighbourhood of target words when model_ver='dm'
        (i.e. how many words left and right are regarded as context). When
        model_ver='dm' context_size has to greater than 0, when
        model_ver='dbow' context_size has to be 0.

    num_noise_words: int
        Number of noise words to sample from the noise distribution.

    vec_dim: int
        Dimensionality of vectors to be learned (for paragraphs and words).

    num_epochs: int
        Number of iterations to train the model (i.e. number
        of times every example is seen during training).

    batch_size: int
        Number of examples per single gradient update.

    lr: float
        Learning rate of the Adam optimizer.

    save_all: bool, default=False
        Indicates whether a checkpoint is saved after each epoch.
        If false, only the best performing model is saved.

    generate_plot: bool, default=True
        Indicates whether a diagnostic plot displaying loss value over
        epochs is generated after each epoch.

    max_generated_batches: int, default=5
        Maximum number of pre-generated batches.

    num_workers: int, default=1
        Number of batch generator jobs to run in parallel. If value is set
        to -1 number of machine cores are used.
    """
    if model_ver not in ('dm', 'dmspline', 'dbow'):
        raise ValueError("Invalid version of the model")

    model_ver_is_dbow = model_ver == 'dbow'
    model_ver_is_dm = model_ver == 'dm'
    model_ver_is_dmspline = model_ver == 'dmspline'

    if model_ver_is_dbow and context_size != 0:
        raise ValueError("Context size has to be zero when using dbow")
    if not model_ver_is_dbow:
        if vec_combine_method not in ('sum', 'concat'):
            raise ValueError("Invalid method for combining paragraph and word "
                             "vectors when using dm")
        if context_size <= 0:
            raise ValueError("Context size must be positive when using dm")

    # dataset = load_dataset(data_file_name, model_ver)
    # nce_data = NCEData(
    #     dataset,
    #     batch_size,
    #     context_size,
    #     num_noise_words,
    #     max_generated_batches,
    #     num_workers,
    #     model_ver)
    # nce_data.start()

    print ('Loading data and making data loader ...')
    doc_ids, context_ids, target_noise_ids, word_to_ind_dict = load_and_cache_data(data_file_root=data_file_name, num_context_words=context_size, num_noise_words=num_noise_words)
    dataloader = make_dataloader((doc_ids, context_ids, target_noise_ids), batch_size)

    all_doc_ids = set()
    for i in doc_ids.tolist():
        all_doc_ids.add(i)

    print ('num unique doc ids: ', len(all_doc_ids))

    try:
        _run(dataloader, data_file_name, all_doc_ids,
             word_to_ind_dict, context_size, num_noise_words, vec_dim,
             num_epochs, batch_size, lr, model_ver, vec_combine_method,
             save_all, generate_plot, model_ver_is_dbow, model_ver_is_dm)
    except KeyboardInterrupt:
        nce_data.stop()


def _run(dataloader,
         data_file_name,
         all_doc_ids,
         word_to_ind_dict,
         context_size,
         num_noise_words,
         vec_dim,
         num_epochs,
         batch_size,
         lr,
         model_ver,
         vec_combine_method,
         save_all,
         generate_plot,
         model_ver_is_dbow,
         model_ver_is_dm):

    vocabulary_size = len(word_to_ind_dict)
    print ('vocab size: ', vocabulary_size)

    if model_ver_is_dbow:
        model = DBOW(vec_dim=vec_dim, num_docs=len(all_doc_ids), num_words=vocabulary_size)
        cost_func = NegativeSampling()
    elif model_ver_is_dm:
        model = DM(vec_dim=vec_dim, num_docs=len(all_doc_ids), num_words=vocabulary_size)
        cost_func = NegativeSampling()
    else:
        print 'Initializing spline model'
        model = DMSpline(vec_dim, num_splines=len(all_doc_ids), num_words=vocabulary_size)
        cost_func = NegativeSamplingWithSpline()

    # Only apply weight decay to the offset vectors
    params_to_decay = []
    other_params = []
    for name, param in model.named_parameters():
        if name == '_D':
            params_to_decay.append(param)
        else:
            other_params.append(param)
    
    optimizer1 = Adam(params=params_to_decay, lr=lr, weight_decay=0.01)
    optimizer2 = Adam(params=other_params, lr=lr)

    if torch.cuda.is_available():
        model.cuda()

    num_batches = len(dataloader)
    num_docs = len(all_doc_ids)
    print("Dataset comprised of {:d} documents.".format(num_docs))
    print ("Num batches: ", num_batches)
    print("Vocabulary size is {:d}.\n".format(vocabulary_size))
    print("Training started.")

    # print('num batches: ', num_batches)
    # exit()
    # num_epochs = 1
    # num_batches = 5

    best_loss = float("inf")
    prev_model_file_path = None

    for epoch_i in range(num_epochs):
        epoch_start_time = time.time()
        loss = []

        ind_to_word_dict = {}
        for w in word_to_ind_dict:
            ind_to_word_dict[word_to_ind_dict[w]] = w

        for batch_i, batch in enumerate(dataloader):
            # print 'curr batch: ', batch_i
            curr_doc_ids, curr_context_ids, curr_target_noise_ids = batch
            if torch.cuda.is_available():
                curr_doc_ids = curr_doc_ids.cuda()
                curr_context_ids = curr_context_ids.cuda()
                curr_target_noise_ids = curr_target_noise_ids.cuda()

            # curr_ind = 0
            # batch_row = curr_context_words[curr_ind]
            # for val in range(0, len(curr_context_ids)):
            #     for val2 in range(0, len(curr_context_ids[val])):
            #         print (curr_context_ids[val][val2].item(), ind_to_word_dict[curr_context_ids[val][val2].item()])
            #     print ('doc id: ', curr_doc_ids[val])
            # exit()

            if model_ver_is_dbow:
                x = model.forward(batch.doc_ids, batch.target_noise_ids)

            elif model_ver_is_dm:
                x = model.forward(
                    curr_context_ids,
                    curr_doc_ids,
                    curr_target_noise_ids)
                x = cost_func.forward(x)

            else:
                x, D  = model.forward(
                    batch.x_vals,
                    batch.doc_ids,
                    batch.context_ids,
                    batch.target_noise_ids)
                x = cost_func.forward(x, D, batch.doc_ids)

            loss.append(x.item())
            model.zero_grad()
            x.backward()
            optimizer1.step()
            optimizer2.step()
            _print_progress(epoch_i, batch_i, num_batches)
            # break

        # end of epoch
        loss = torch.mean(torch.FloatTensor(loss))
        is_best_loss = loss < best_loss
        best_loss = min(loss, best_loss)

        print ('loss: ', loss, 'best_loss: ', best_loss, 'is_best_loss: ', is_best_loss)

        state = {
            'epoch': epoch_i + 1,
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer1_state_dict': optimizer1.state_dict(),
            'optimizer2_state_dict': optimizer2.state_dict(),
            'word_to_index_dict': word_to_ind_dict
        }

        prev_model_file_path = save_training_state(
            data_file_name,
            model_ver,
            vec_combine_method,
            context_size,
            num_noise_words,
            vec_dim,
            num_docs,
            vocabulary_size,
            batch_size,
            lr,
            epoch_i,
            loss,
            state,
            save_all,
            generate_plot,
            is_best_loss,
            prev_model_file_path,
            model_ver_is_dbow)

        epoch_total_time = round(time.time() - epoch_start_time)
        print(" ({:f}s) - loss: {:.4f}".format(epoch_total_time, loss))


def _print_progress(epoch_i, batch_i, num_batches):
    progress = (float((batch_i + 1)) / num_batches) * 100

    if batch_i % 500 == 0:
        print("Epoch {}, batch {}: {}%".format(epoch_i + 1, batch_i, progress))


if __name__ == '__main__':
    fire.Fire()

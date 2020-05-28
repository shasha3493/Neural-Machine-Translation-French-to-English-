'''Functions related to training and testing.'''

import torch
import a2_bleu_score


from tqdm import tqdm


def train_for_epoch(model, dataloader, optimizer, device):
    '''Train an EncoderDecoder for an epoch

    An epoch is one full loop through the training data. This function:

    1. Defines a loss function using :class:`torch.nn.CrossEntropyLoss`,
       keeping track of what id the loss considers "padding"
    2. For every iteration of the `dataloader` (which yields triples
       ``F, F_lens, E``)
       1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same
          for ``F_lens`` and ``E``.
       2. Zeros out the model's previous gradient with ``optimizer.zero_grad()``
       3. Calls ``logits = model(F, F_lens, E)`` to determine next-token
          probabilities.
       4. Modifies ``E`` for the loss function, getting rid of a token and
          replacing excess end-of-sequence tokens with padding using
        ``model.get_target_padding_mask()`` and ``torch.masked_fill``
       5. Flattens out the sequence dimension into the batch dimension of both
          ``logits`` and ``E``
       6. Calls ``loss = loss_fn(logits, E)`` to calculate the batch loss
       7. Calls ``loss.backward()`` to backpropagate gradients through
          ``model``
       8. Calls ``optim.step()`` to update model parameters
    3. Returns the average loss over sequences

    Parameters
    ----------
    model : EncoderDecoder
        The model we're training.
    dataloader : HansardDataLoader
        Serves up batches of data.
    device : torch.device
        A torch device, like 'cpu' or 'cuda'. Where to perform computations.
    optimizer : torch.optim.Optimizer
        Implements some algorithm for updating parameters using gradient
        calculations.

    Returns
    -------
    avg_loss : float
        The total loss divided by the total numer of tokens
    '''    
    n_tokens = 0
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index = -1)
    for F, F_lens, E in tqdm(dataloader):
        n_tokens += torch.sum(F_lens)
        F, F_lens, E = F.to(device), F_lens.to(device), E.to(device)
        optimizer.zero_grad()
        logits = model(F, F_lens, E)
        E = E[1:,:] # Removing the initial SOS token
        pad_mask = model.get_target_padding_mask(E)

        # -1 will be the ignore index parameter of loss function
        
        E = E.masked_fill(pad_mask, -1)
        logits = torch.flatten(logits, end_dim = 1)
        E = torch.flatten(E, end_dim = 1)
        loss = 0
        loss = criterion(logits, E)
        total_loss += loss.item()*E.shape[0]
        loss.backward()
        optimizer.step()
        del F, F_lens, E, logits, loss
        torch.cuda.empty_cache()

    avg_loss = total_loss/n_tokens

    return avg_loss



def compute_batch_total_bleu(E_ref, E_cand, target_sos, target_eos):
    '''Compute the total BLEU score over elements in a batch

    Parameters
    ----------
    E_ref : torch.LongTensor
        A batch of reference transcripts of shape ``(T, N)``, including
        start-of-sequence tags and right-padded with end-of-sequence tags.
    E_cand : torch.LongTensor
        A batch of candidate transcripts of shape ``(T', N)``, also including
        start-of-sequence and end-of-sequence tags.
    target_sos : int
        The ID of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The ID of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    total_bleu : float
        The sum total BLEU score for across all elements in the batch. Use
        n-gram precision 4.
    '''

    total_bleu = 0
    N = E_ref.shape[1]
    
    for i in range(N):
        cand = E_cand[:,i].tolist()
        ref = E_ref[:,i].tolist()
        while target_eos in cand:
            _ = cand.pop(cand.index(target_eos))
        while target_sos in cand:
            _ = cand.pop(cand.index(target_sos))
        while target_eos in ref:
            _ = ref.pop(ref.index(target_eos))
        while target_sos in ref:
            _ = ref.pop(ref.index(target_sos))
        total_bleu += a2_bleu_score.BLEU_score(ref, cand, 4)

    return(total_bleu)



def compute_average_bleu_over_dataset(
        model, dataloader, target_sos, target_eos, device):
    '''Determine the average BLEU score across sequences

    This function computes the average BLEU score across all sequences in
    a single loop through the `dataloader`.

    1. For every iteration of the `dataloader` (which yields triples
       ``F, F_lens, E_ref``):
       1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same
          for ``F_lens``. No need for ``E_cand``, since it will always be
          compared on the CPU.
       2. Performs a beam search by calling ``b_1 = model(F, F_lens)``
       3. Extracts the top path per beam as ``E_cand = b_1[..., 0]``
       4. Computes the total BLEU score of the batch using
          :func:`compute_batch_total_bleu`
    2. Returns the average per-sequence BLEU score

    Parameters
    ----------
    model : EncoderDecoder
        The model we're testing.
    dataloader : HansardDataLoader
        Serves up batches of data.
    target_sos : int
        The ID of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The ID of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    avg_bleu : float
        The total BLEU score summed over all sequences divided by the number of
        sequences
    '''
    total_bleu = 0
    n_seq = 0

    for F, F_lens, E_ref in dataloader:

        n_seq += len(F_lens)
        F, F_lens = F.to(device), F_lens.to(device)
        b_1 = model(F, F_lens)
        E_cand = b_1[..., 0]
        total_bleu += compute_batch_total_bleu(E_ref, E_cand, target_sos, target_eos)
        del F, F_lens, E_ref
        torch.cuda.empty_cache()

    avg_bleu = total_bleu/n_seq

    return avg_bleu

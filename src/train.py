# pylint: disable=no-member

import torch
import torch.nn as nn
import torch.optim as optim

from configs import args
from dataset import ReviewDataset

from model import ReviewPerceptronClassifier
from tqdm import tqdm

from utils import set_seed_everywhere, handle_dirs, make_train_state, \
    generate_batches, compute_accuracy, update_train_state
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir,
                                            args.vectorizer_file)

        args.model_state_file = os.path.join(args.save_dir,
                                             args.model_state_file)

        print("Expanded filepaths: ")
        print("\t{}".format(args.vectorizer_file))
        print("\t{}".format(args.model_state_file))

    # Check CUDA
    if not torch.cuda.is_available():
        args.cuda = False

    print("Using CUDA: {}".format(args.cuda))

    args.device = torch.device("cuda" if args.cuda else "cpu")

    set_seed_everywhere(args.seed, args.cuda)

    #Initializations

    # if args.reload_from_files:
    #     # training from a checkpoint
    #     print("Loading dataset and vectorizer")
    #     dataset = ReviewDataset.load_dataset_and_load_vectorizer(args.review_csv,
    #                                                              args.vectorizer_file)
    # else:
    #     print("Loading dataset and creating vectorizer")
    #     # create dataset and vectorizer
    #     dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv)
    #     dataset.save_vectorizer(args.vectorizer_file)

    # vectorizer = dataset.get_vectorizer()

    # classifier = ReviewPerceptronClassifier(num_features=len(vectorizer.review_vocab), num_classes=1)

    dataset = args.dataset
    vectorizer = args.vectorizer
    classifier = args.classifier

    classifier = classifier.to(args.device)

    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     mode='min', factor=0.5,
                                                     patience=1)

    train_state = make_train_state(args)


    epoch_bar = tqdm(desc='training routine',
                     total=args.num_epochs,
                     position=0)

    dataset.set_split('train')
    train_bar = tqdm(desc='split=train',
                     total=dataset.get_num_batches(args.batch_size),
                     position=1,
                     leave=True)
    dataset.set_split('val')
    val_bar = tqdm(desc='split=val',
                   total=dataset.get_num_batches(args.batch_size),
                   position=1,
                   leave=True)
    dataset.set_split('test')
    test_bar = tqdm(desc='split=test',
                   total=dataset.get_num_batches(args.batch_size),
                   position=1,
                   leave=True)
    if args.train:    
        try:        
            for epoch_index in range(args.num_epochs):
                train_state['epoch_index'] = epoch_index

                # Iterate over training dataset

                # setup: batch generator, set loss and acc to 0, set train mode on
                dataset.set_split('train')
                batch_generator = generate_batches(dataset,
                                                batch_size=args.batch_size,
                                                device=args.device)
                running_loss = 0.0
                running_acc = 0.0
                classifier.train()

                for batch_index, batch_dict in enumerate(batch_generator):
                    # the training routine is these 5 steps:

                    # --------------------------------------
                    # step 1. zero the gradients
                    optimizer.zero_grad()

                    # step 2. compute the output
                    # print(f'Classifier is \n {classifier}')
                    if args.architecture_type == 'RNN':
                        y_pred = classifier(x_in=batch_dict['x_data'].float(), x_lengths=batch_dict['vector_len'])
                    else:
                        y_pred = classifier(x_in=batch_dict['x_data'].float())
                    y_true = batch_dict['y_target']

                    # step 3. compute the loss
                    # print(f'y_pred shape is {y_pred.size()}')
                    # print(f'y_true shape is {y_true.size()}')

                    loss = loss_func(y_pred, batch_dict['y_target'].float())
                    loss_t = loss.item()
                    running_loss += (loss_t - running_loss) / (batch_index + 1)

                    # step 4. use loss to produce gradients
                    loss.backward()

                    # step 5. use optimizer to take gradient step
                    optimizer.step()
                    # -----------------------------------------
                    # compute the accuracy
                    acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
                    running_acc += (acc_t - running_acc) / (batch_index + 1)

                    # update bar
                    train_bar.set_postfix(loss=running_loss,
                                        acc=running_acc,
                                        epoch=epoch_index)
                    train_bar.update()

                train_state['train_loss'].append(running_loss)
                train_state['train_acc'].append(running_acc)

                # Iterate over val dataset

                # setup: batch generator, set loss and acc to 0; set eval mode on
                dataset.set_split('val')
                batch_generator = generate_batches(dataset,
                                                batch_size=args.batch_size,
                                                device=args.device)
                running_loss = 0.
                running_acc = 0.
                classifier.eval()

                for batch_index, batch_dict in enumerate(batch_generator):
                    # compute the output
                    if args.architecture_type == 'RNN':
                        y_pred = classifier(x_in=batch_dict['x_data'].float(), x_lengths=batch_dict['vector_len'])
                    else:
                        y_pred = classifier(x_in=batch_dict['x_data'].float())
                    
                    # step 3. compute the loss
                    loss = loss_func(y_pred, batch_dict['y_target'].float())
                    loss_t = loss.item()
                    running_loss += (loss_t - running_loss) / (batch_index + 1)

                    # compute the accuracy
                    acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
                    running_acc += (acc_t - running_acc) / (batch_index + 1)

                    val_bar.set_postfix(loss=running_loss,
                                        acc=running_acc,
                                        epoch=epoch_index)
                    val_bar.update()

                train_state['val_loss'].append(running_loss)
                train_state['val_acc'].append(running_acc)

                train_state = update_train_state(args=args, model=classifier,
                                                train_state=train_state)

                scheduler.step(train_state['val_loss'][-1])

                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()

                if train_state['stop_early']:
                    break

                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()

        except KeyboardInterrupt:
            print("Exiting loop")

    # compute the loss & accuracy on the test set using the best available model
    print('Computing the accuracy on the Test set')
    classifier.load_state_dict(torch.load(train_state['model_filename']))
    classifier = classifier.to(args.device)

    dataset.set_split('test')
    batch_generator = generate_batches(dataset,
                                        batch_size=args.batch_size,
                                        device=args.device)
    running_loss = 0.
    running_acc = 0.
    classifier.eval()

    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output
        if args.architecture_type == 'RNN':
            y_pred = classifier(x_in=batch_dict['x_data'].float(), x_lengths=batch_dict['vector_len'])
        else:
            y_pred = classifier(x_in=batch_dict['x_data'].float())
    
        # compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'].float())
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # compute the accuracy
        acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)
        
        test_bar.set_postfix(loss=running_loss,
                                        acc=running_acc,
                                        )
        test_bar.update()

    train_state['test_loss'] = running_loss
    train_state['test_acc'] = running_acc

    print("Test loss: {:.3f}".format(train_state['test_loss']))
    print("Test Accuracy: {:.2f}".format(train_state['test_acc']))




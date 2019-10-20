import torch

def wrapper(param, data_x, data_y, lr_gamma, hidden_dim, layers):

    device = param['device']
    input_dim = param['region_n']
    n_epochs = param['n_epochs']
    outputfolder = param['outputfolder']
    minmax_y = param['minmax_y']

    output_dim = 1  # Output dimension
    drop_prob = 0.5  # Drop probability during training
    train_num = 560
    valid_num = 80
    total_num = len(data_y)

    train_x_tensor, train_y_tensor = get_tensor(
        device, data_x, data_y, 0, train_num)
    valid_x_tensor, valid_y_tensor = get_tensor(
        device, data_x, data_y, train_num+1, train_num+valid_num)

    out_fname = f'hidden_dim_{hidden_dim}_layers_{layers}_lr_gamma_{lr_gamma}'

    mynet = RNNClassifier(
        input_dim, hidden_dim, output_dim, layers, drop_prob).to(device)

    start = time.time()  # Start Learning
    print("Start Learning " + out_fname)

    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(mynet.parameters(), lr=0.001)
    lr_sche = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=lr_gamma)
    train_loss_list = []
    valid_loss_list = []
    for i in range(n_epochs):
        # Train
        mynet.train()
        lr_sche.step()
        outputs = mynet(train_x_tensor)
        loss = criterion(outputs, train_y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_list.append(loss.item())

        # Validation
        mynet.eval()
        valid_result = mynet(valid_x_tensor)
        valid_loss = criterion(valid_result, valid_y_tensor)
        valid_loss_list.append(valid_loss.item())
    train_loss_arr = np.array(train_loss_list)
    valid_loss_arr = np.array(valid_loss_list)

    output_path = f'./figs/{outputfolder}'
    safe_make_dir(output_path)
    # Write loss values in csv file
    write_loss(train_loss_arr, valid_loss_arr, output_path, out_fname)

    # Plot train and validation losses
    plot_train_val_loss(
        n_epochs, train_loss_arr, valid_loss_arr, output_path, out_fname,
        linewidth=0.1, dpi=800, yscale='log', ylim=[0.0001, 10])

    end = time.time()  # Learning Done
    print(f"Learning Done in {end-start}s")

    # Test
    mynet.eval()
    with torch.no_grad():
        test_x_tensor, test_y_tensor = get_tensor(
            device, data_x, data_y, train_num+valid_num+1, total_num)

        test_result = mynet(test_x_tensor)
        test_loss = criterion(test_result, test_y_tensor)
        print(f"Test Loss: {test_loss.item()}")
    plot_result(
        test_y_tensor, test_result, minmax_y, output_path, out_fname)

criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(mynet.parameters(), lr=0.001)
lr_sche = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=100, gamma=lr_gamma)

param = {'gamma_list': [0.99, 0.975, 0.95],
         'hidden_dim_list': [200, 300],
         'layers_list': [3, 4]}

product_set = itertools.product(
        param['gamma_list'],
        param['hidden_dim_list'],
        param['layers_list'])

for lr_ gamma, hidden_dim, layers in product_set:
    wrapper(param, data_x, data_y, lr_gamma, hidden_dim, layers)
import agnes.common.tests as test_pack


if __name__ == '__main__':
    print("Testing is started...")
    test_pack.MLP_Discrete.test_single()
    test_pack.MLP_Continuous.test_single()
    test_pack.MLP_Continuous.test_vec()
    test_pack.CNN_Discrete.test_single_cnn()
    test_pack.CNN_Discrete.test_single_cnn_rnn()
    test_pack.CNN_Discrete.test_single_cnn_a2c()
    test_pack.CNN_Discrete.test_single_cnn_ppo_rnd()
    test_pack.RNN_Discrete.test_single()
    test_pack.MPI_runner.test_single()
    print("Tests successfully completed!")

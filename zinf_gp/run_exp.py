from metrics import fixed_top_X


def run_model(svi_file=None,):

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--time', type=str, help="Temporal division of data",
                        choices=['qtr' ,'biannual', 'annual'], default='qtr')
    parser.add_argument('--data_path', type=str, help="Path to opioid data",
                        default='/cluster/tufts/hugheslab/datasets/NSF_OD/')
    parser.add_argument('--kernel', type=str, help="How to make kernels",
                        choices=['st_only', 'svi_only', 'svi_full'] ,)
    parser.add_argument('--auto_kernel', action='store_true', help="If present, add a kernel with autoregressive features.")
    parser.add_argument('--inducing_points', type=int, required=True, default=200,
                        help="Number of inducing points to use")
    parser.add_argument('--samples', type=int, default=10,
                        help="Number of inducing points to use")
    parser.add_argument('--iterations', type=int, required=True,
                        help="Number of iterations to run")
    parser.add_argument('--seed', type=int, default=1,
                        help="seed to use for inducing points")
    parser.add_argument('--learning_rate', type=float,
                        help="Adam LR", default=0.005)
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save results in')


    args = parser.parse_args()

    run_model(**vars(args))

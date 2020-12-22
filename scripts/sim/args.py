import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument('load', nargs='?', help='load pkl')
parser.add_argument('--plot', action='store_true',help='plot and save')
parser.add_argument('--show', action='store_true', help='show plot')
parser.add_argument('--wait', action='store_true', help='wait between iterations')
parser.add_argument('--seed', type=int, help='rng seed')
parser.add_argument('--step', default=100, type=int, help='additional samples per step')
parser.add_argument('--n-init', default=500, type=int, help='initial sample size')
parser.add_argument('--e-init', default=1, type=int, help='epsilon init')
parser.add_argument('--o-init', default=0., type=int, help='omega init')
parser.add_argument('--n-iter', '-N', default=10, type=int, help='number of iterations')
parser.add_argument('--n-src', default=2, type=int, help='number of sources')
parser.add_argument('--o-coef', default=1, type=int, help='omega = birth + lips * eps / o_coef')
parser.add_argument('--pre-it', default=5, type=float, help='number of iterations before eps/omega birth skip')
parser.add_argument('--e-coef', default=0.9, type=float, help='new_eps *= e_coef')# * o_coef * (max_dist - birth) / lips')
parser.add_argument('--lips', type=float, help='precomputed lipschitz constant')
parser.add_argument('--stop', default=10, help='number of random trials before an early stop')
parser.add_argument('--len', '-l', type=float, nargs=2, default=[1.,1.], help='arm lengths')
parser.add_argument('--fun', default='min', choices=['sum', 'min'],
                    help='sum: sum of distances to sources\nmin: min distance to source')
parser.add_argument('--mode', default='body', choices=['head', 'body'],
                    help='function computed robot body or head')

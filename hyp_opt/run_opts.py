from multiprocessing.spawn import freeze_support

from hyp_opt import de_hyp_opt, brkga_hyp_opt, multipop_hyp_opt

if __name__ == '__main__':
    freeze_support()
    # de_hyp_opt.main()
    brkga_hyp_opt.main()
    # multipop_hyp_opt.main()

    # ga_hyp_opt.main()

import argparse




def get_args():
    parser = argparse.ArgumentParser()
    
    # Dataset arguments 
    parser.add_argument("--dataset_dir" , type= str , default="./datasets")
    parser.add_argument("--dataset_mode" , type = str , default = "train" , help = "train or test" )

    # Image related arguments (Reference Person Image)
    parser.add_argument("--load_height" , type = int , default = 1024 , help="Heigth of refrence input imgae")
    parser.add_argument("--load-width" , type = int , default = 768 , help = "Width of reference input image")

    
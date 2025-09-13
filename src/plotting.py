import matplotlib.pyplot as plt, os

def save_line(x,y,title,fname,out_dir):
    plt.figure(figsize=(10,4))
    plt.plot(x,y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,fname))
    plt.close()

def scatter_and_save(x,y,title,fname,out_dir):
    plt.figure(figsize=(6,4))
    plt.scatter(x,y,alpha=0.6)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,fname))
    plt.close()

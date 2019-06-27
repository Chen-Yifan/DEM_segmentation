import os 
  
# Function to rename multiple files 
def main(): 
    directories = ['annotations_128_overlap']
    PATH = "/home/yifanc3/dataset/data/"
    i = 0
    for d in directories:
        file = os.path.join(PATH, d)
        print(file)
        for filename in os.listdir(file): 
            dst = file+'/'+filename[-11:]
            src = file+'/'+filename
            print(src,dst)
            os.rename(src, dst) 
            i += 1
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 


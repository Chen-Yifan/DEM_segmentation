import os 
  
# Function to rename multiple files 
def main(): 
    i = 0
    PATH = "/home/yifanc3/dataset/data/frames_128/"
    for filename in os.listdir(PATH): 
        dst = PATH+filename[-11:]
        src = PATH+filename
        print(src,dst)
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i += 1
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 


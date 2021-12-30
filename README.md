# Object-detection-using-EdgeBoxes-and-Selective-Search_Algorithm
A comparative study between two algorithms namely selective search and edgeboxes for object detection


Code Execution:

    1. Go to the project folder.
    2. Right click there and click open the folder in terminal.
    3. Type the following commands in terminal-> 
        for selective search: 
          for color strategy: 
            python selective_search.py ./JPEGImages/1.jpg ./Annotations/1.xml color 
          for all strategy: 
            python selective_search.py ./JPEGImages/1.jpg ./Annotations/1.xml all 
        for edgeboxes: 
          python edgeboxes.py ./JPEGImages/1.jpg ./Annotations/1.xml
    4. You can change the images by changing the number from 1 to 8 in both JPEGImages and Annotations.
    5. After running the command press windows key to see the detected objects thrice.
    6. You can view the comparisons in the Results.pdf file.

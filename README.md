# Group-movie-recommender-system
In this project, we create a matrix factorization based Recommender System for a group of users.
We try out 3 different methods.
<li> After Factorization: where we aggregate factors of users into group factors after factorization </li>
<li> Before Factorization(BF): where we aggregate users' ratings into a virtual user. </li>
<li> Weighted Before Factorization(WBF): Same as BF except that no. of movies watched by users are taken
    as weights. We solve it using ridge regression method. </li>

## Paper
Project is based on the following paper:

http://www.sciencedirect.com/science/article/pii/S0020025516300196

## Dataset
Dataset: https://grouplens.org/datasets/movielens/100k/

## Video
Video: https://www.youtube.com/watch?v=ycf2sY2XnN8

## Execution Dependencies
The notebook can be run directly.
Dataset is included in the github repo.
Also, the python code can be run by: 
```
python ./GroupRec.py
```

We are using pandas, numpy, scipy and warnings modules. Install them by
running.
```
pip install numpy
pip install pandas
pip install scipy
```
The arguments are taken via a config file config.conf that is present in the
same folder. The hyperparameters for matrix factorization, group sizes and
no. of generated groups can be changed through the config file.


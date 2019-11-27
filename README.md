# Code for the CI Workshop 2019 submission with the title "Algorithm Selection as Recommendation: From Collaborative Filtering to Dyad Ranking".

This repository holds the code for the CI Workshop 2019 submission with the title "Algorithm Selection as Recommendation: 
From Collaborative Filtering to Dyad Ranking" by Alexander Tornede, Marcel Wever and Eyke Hüllermeier. 

## Reproduction of Results
In order to reproduce the results, we assume you to have a MySQL database server running. Into this database, you have to import the three tables from the folder "db" which are stored in the form of SQL dumps. After this was done, you have to adapt four settings within the class "ci.workshop.experiments.evaluator.ExperimentRunner". In line 46, you have to adapt the database settings according to your database. The first string is the URL of the server, the second one is the username which should be used by the program and the third is the respective password. Furthermore, you have to adapt the database name in line 29 to the name of the database where you imported the three tables. After having performed these steps, you should be able to run the main-method of the class. Doing so, will run all experiments and store the associated results in a table entitled by the string given in line 30.  

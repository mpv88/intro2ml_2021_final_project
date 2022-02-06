# i2ml_2021_final_project
twitter related prediction problems

Deliver a single document (as a pdf file, written in English or Italian), within the exam date, by email, to the teacher. The document maximum length is fixed at 4 pages (excluding references), if the document is drafted according to this LaTex template, or 1200 words (including every word: title, authors, references, …), otherwise.
https://www.overleaf.com/project/61df5902f8b32afb02235e6e

The document will contain (not necessarily in the following order, not necessarily in a structure of sections corresponding to the following items):
    0) the problem statement, that is a formal description of what is the information (input) upon which a decision/figure (output) has to be made/computed
    1) a description of one or more performance indexes able to capture the degree to which a(ny) solution solves the problem, or some other evaluation criteria
    2) a description of the proposed solution, from the algorithmic point of view, with zero or scarce focus on the tools used to implement it
    3) a description of the experimental evaluation of the solution, including
        -a description of the used data, if any
        -a description of the experimental procedure and the comparison baseline, if any
        -the presentation and the discussion of the results

The students are allowed (and encouraged) to refer to existing (technical/scientific/research) literature for shrinking/omitting obvious parts of the description, if appropriate. The students are not required to deliver any source code or other material. However, if needed for gaining more insights on their work, the students will be asked to provide more material or to answer some questions. If the project has been done by a group of students, the document must show, for each student of the group, which (at least one) of the following activities the student took part in:
   - problem statement
   - solution design
   - solution development
   - data collection
   - writing

# Twitter food popularity
The goal is to build a tool for predicting how popular will be a tweet about food. 
There is no data available for this problem: the student is hence required to obtain it, if the proposed solution is based on data (likely it will be!).

Sources:
http://belindazeng.github.io/goingviral/
https://towardsdatascience.com/using-data-science-to-predict-viral-tweets-615b0acc2e1e
https://towardsdatascience.com/extracting-data-from-twitter-using-python-5ab67bff553a
https://www.sciencedirect.com/science/article/abs/pii/S0957417420306096

#TODO:
a) plot OBB vs n_estimators/accuracy vs n_estimators 
https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html#sphx-glr-auto-examples-ensemble-plot-ensemble-oob-py
https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html
https://datascience.stackexchange.com/questions/86049/how-plot-gridsearch-results
https://datascience.stackexchange.com/questions/64441/how-to-interpret-classification-report-of-scikit-learn
for ROC the auc of the random model is 0.5.
for PR curve the auc of the random model is n_positive/(n_positive+n_negative).
Area under ROC curve is very useful metric to validate classification model because it is threshold and scale invariant while accuracy depends on the chosen threshold value and is scale variant
https://medium.com/nerd-for-tech/accuracy-vs-auc-roc-a8e7a384d153
Accuracy+AUCROC & F1+AUCPR
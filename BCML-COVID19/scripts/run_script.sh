ver1=4.1_sdoh_lymphocytesabs
out1=death
ver2=4.0_sdoh_lymphocytesabs
out2=vent

#python3 train_all.py --version $ver1 --outcome $out1 --model logreg --impute mean
#python3 train_all.py --version $ver1 --outcome $out1 --model logreg --impute median
#python3 train_all.py --version $ver1 --outcome $out1 --model logreg --impute mice
#python3 train_all.py --version $ver1 --outcome $out1 --model logreg --impute mice+smote

#python3 train_all.py --version $ver1 --outcome $out1 --model xgboost --impute mean
#python3 train_all.py --version $ver1 --outcome $out1 --model xgboost --impute median
#python3 train_all.py --version $ver1 --outcome $out1 --model xgboost --impute mice
#python3 train_all.py --version $ver1 --outcome $out1 --model xgboost --impute mice+smote

#python3 train_all.py --version $ver1 --outcome $out1 --model rf --impute mean
#python3 train_all.py --version $ver1 --outcome $out1 --model rf --impute median
#python3 train_all.py --version $ver1 --outcome $out1 --model rf --impute mice
#python3 train_all.py --version $ver1 --outcome $out1 --model rf --impute mice+smote

#python3 train_all.py --version $ver1 --outcome $out1 --model gp --impute mean
#python3 train_all.py --version $ver1 --outcome $out1 --model gp --impute median
#python3 train_all.py --version $ver1 --outcome $out1 --model gp --impute mice
#python3 train_all.py --version $ver1 --outcome $out1 --model gp --impute mice+smote

#python3 train_all.py --version $ver1 --outcome $out1 --model xgboostrf --impute mean
#python3 train_all.py --version $ver1 --outcome $out1 --model xgboostrf --impute median
#python3 train_all.py --version $ver1 --outcome $out1 --model xgboostrf --impute mice
#python3 train_all.py --version $ver1 --outcome $out1 --model xgboostrf --impute mice+smote

#python3 train_all.py --version $ver1 --outcome $out1 --model mlp --impute mean
#python3 train_all.py --version $ver1 --outcome $out1 --model mlp --impute median
#python3 train_all.py --version $ver1 --outcome $out1 --model mlp --impute mice
#python3 train_all.py --version $ver1 --outcome $out1 --model mlp --impute mice+smote


#python3 train_all.py --version $ver2 --outcome $out2 --model logreg --impute mean
#python3 train_all.py --version $ver2 --outcome $out2 --model logreg --impute median
#python3 train_all.py --version $ver2 --outcome $out2 --model logreg --impute mice
#python3 train_all.py --version $ver2 --outcome $out2 --model logreg --impute mice+smote

#python3 train_all.py --version $ver2 --outcome $out2 --model xgboost --impute mean
python3 train_all.py --version $ver2 --outcome $out2 --model xgboost --impute median --n_trials 500
#python3 train_all.py --version $ver2 --outcome $out2 --model xgboost --impute mice
#python3 train_all.py --version $ver2 --outcome $out2 --model xgboost --impute mice+smote

#python3 train_all.py --version $ver2 --outcome $out2 --model rf --impute mean
#python3 train_all.py --version $ver2 --outcome $out2 --model rf --impute median
#python3 train_all.py --version $ver2 --outcome $out2 --model rf --impute mice
#python3 train_all.py --version $ver2 --outcome $out2 --model rf --impute mice+smote

#python3 train_all.py --version $ver2 --outcome $out2 --model gp --impute mean
#python3 train_all.py --version $ver2 --outcome $out2 --model gp --impute median
#python3 train_all.py --version $ver2 --outcome $out2 --model gp --impute mice
#python3 train_all.py --version $ver2 --outcome $out2 --model gp --impute mice+smote

#python3 train_all.py --version $ver2 --outcome $out2 --model xgboostrf --impute mean
#python3 train_all.py --version $ver2 --outcome $out2 --model xgboostrf --impute median
#python3 train_all.py --version $ver2 --outcome $out2 --model xgboostrf --impute mice
#python3 train_all.py --version $ver2 --outcome $out2 --model xgboostrf --impute mice+smote

#python3 train_all.py --version $ver2 --outcome $out2 --model mlp --impute mean
#python3 train_all.py --version $ver2 --outcome $out2 --model mlp --impute median
#python3 train_all.py --version $ver2 --outcome $out2 --model mlp --impute mice
#python3 train_all.py --version $ver2 --outcome $out2 --model mlp --impute mice+smote

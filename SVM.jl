include("./first_code.jl")

# SVC du prof, ca marche mais on peut pas avoir les proba
mach_SVM_d = machine(SVC(probability = true), training_dropped_x, training_dropped_y) |> fit!

pred_SVM_d = predict(mach_SVM_d, training_dropped_x)
# predict_mode(mach_SVM_d, training_dropped_x) --> NE MARCHE PAS






# IGNORER : 

#print(pred_SVM_d)

# proba_SVM_d = predict_mode(mach_SVM_d, training_dropped_x)

#= err_rate_SVM_d = mean(pred_SVM_d .!= training_dropped_y)
print(err_rate_SVM_d) =#


#= mach_SVM_d_2 = svmtrain(training_dropped_x, training_dropped_y)
print(prediction_type(mach_SVM_d_2))

pred_SVM_d_2, decision_values = svmpredict(mach_SVM_d_2, training_dropped_x)

print(pred_SVM_d_2) =#


#= 
# Training data
X_SVM = Matrix(training_dropped_x)'
y_SVM = coerce(training_dropped_y, Continuous) .- 1


# Precomputed matrix for training (corresponds to linear kernel)
K = X_SVM' * X_SVM
typeof(K)
model = svmtrain(K, y_SVM, kernel=Kernel.Precomputed)

# Prediction
ỹ, _ = svmpredict(model, K)

 =#







 

 # Methode que j'ai trouvé
import LIBSVM:svmtrain, SVM, svmpredict

model_for_tuning = svmtrain(Matrix(training_dropped_x)', Vector(training_dropped_y); probability=true, cost = 0.5)
classes, probs = svmpredict(model_for_tuning, Matrix(training_dropped_x)')

proba_true = probs[1, :] # C'est la probabilité d'avoir true pour le train set

err_rate_d = mean(classes .== training_dropped_y) # Enfait, classes donne les résultats inverses


classes_test, probs_test = svmpredict(model_for_tuning, Matrix(test_data)')
probs_test_true = probs_test[1, :]




#= 
tuned_model_SVM = TunedModel(model = model_for_tuning,
                                tuning =  Grid(),
                                resampling = CV(nfolds = 5),
                                range = range(model_for_tuning, :cost, values = [1,2,3]),
                                measure = auc)
tuned_mach_SVM = machine(tuned_model_SVM,
                            training_dropped_x,
                            training_dropped_y) |> fit! =#


include("./first_code.jl")
import LIBSVM:svmtrain, SVM, svmpredict

svm_model_d = svmtrain(Matrix(training_filled_x)', training_filled_y)
pred_test_ = svmpredict(svm_model_d, Matrix(test_data)')
pred_train = svmpredict(svm_model_d, Matrix(training_dropped_x)')

print(nrow(training_dropped_x))
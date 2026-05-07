# Uniform mislabelled

demo_single:
	python -m mislabelled_pipeline --data_name fmnist --data_folder data --subset_ratio 0.1 --model_name resnet20 --model_conf configs/resnet/resnet_2layer_tune_model.json --training_conf configs/resnet/resnet_2layer_tune_train_mnist.json --inf_fn_conf configs/resnet/tracin_resnet.json --error mislabelled_uniform --contamination 0.1 --seed 0 --device cpu
	python -m mislabelled_pipeline --data_name fmnist --data_folder data --subset_ratio 0.1 --model_name resnet20 --model_conf configs/resnet/resnet_2layer_tune_model.json --training_conf configs/resnet/resnet_2layer_tune_train_mnist.json --inf_fn_conf configs/resnet/tracin_resnet.json --error mislabelled_cb --contamination 0.1 --seed 0 --device cpu
	python -m anomaly_pipeline --data_name fmnist --data_folder data --subset_ratio 0.1 --model_name resnet20 --model_conf configs/resnet/resnet_2layer_tune_model.json --training_conf configs/resnet/resnet_2layer_tune_train_fmnist.json --inf_fn_conf configs/resnet/tracin_resnet.json --ood_data_name mnist --contamination 0.1 --seed 0 --device cpu

demo_mixed:
	python -m mixed_errors_pipeline --data_name mnist --data_folder data --subset_ratio 0.1 --model_name resnet20 --model_conf configs/resnet/resnet_2layer_tune_model.json --training_conf configs/resnet/resnet_2layer_tune_train_mnist.json --inf_fn_conf configs/resnet/tracin_resnet.json --seed 0 --device cpu


# Uniform class noise

uniform-class-noise-resnet:
	python -m mislabelled_pipeline --data_name mnist --data_folder data --subset_ratio 0.1 --model_name resnet20 --model_conf configs/resnet/resnet_2layer_tune_model.json --training_conf configs/resnet/resnet_2layer_tune_train_mnist.json --error mislabelled_uniform --contamination 0.1 --seed 0 --device cpu
	python -m mislabelled_pipeline --data_name fmnist --data_folder data --subset_ratio 0.1 --model_name resnet20 --model_conf configs/resnet/resnet_2layer_tune_model.json --training_conf configs/resnet/resnet_2layer_tune_train_fmnist.json --error mislabelled_uniform --contamination 0.1 --seed 0 --device cpu
	python -m mislabelled_pipeline --data_name cifar10 --data_folder data --subset_ratio 0.1 --model_name resnet20 --model_conf configs/resnet/resnet_2layer_tune_model.json --training_conf configs/resnet/resnet_2layer_tune_train_cifar10.json --error mislabelled_uniform --contamination 0.1 --seed 0 --device cpu

uniform-class-noise-vit:
	python -m mislabelled_pipeline --data_name mnist --data_folder data --subset_ratio 0.1 --model_name vit --model_conf configs/vit/vit_2layer_tune_model.json --training_conf configs/vit/vit_2layer_tune_train_mnist.json --inf_fn_conf configs/vit/tracin_vit.json --error mislabelled_uniform --contamination 0.1 --seed 0 --device cpu
	python -m mislabelled_pipeline --data_name fmnist --data_folder data --subset_ratio 0.1 --model_name vit --model_conf configs/vit/vit_2layer_tune_model.json --training_conf configs/vit/vit_2layer_tune_train_fmnist.json --inf_fn_conf configs/vit/tracin_vit.json --error mislabelled_uniform --contamination 0.1 --seed 0 --device cpu
	python -m mislabelled_pipeline --data_name cifar10 --data_folder data --subset_ratio 0.1 --model_name vit --model_conf configs/vit/vit_2layer_tune_model.json --training_conf configs/vit/vit_2layer_tune_train_cifar10.json --inf_fn_conf configs/vit/tracin_vit.json --error mislabelled_uniform --contamination 0.1 --seed 0 --device cpu

uniform-class-noise-convnext:
	python -m mislabelled_pipeline --data_name mnist --data_folder data --subset_ratio 0.1 --model_name convnext --model_conf configs/convnext/convnext_2layer_tune_model.json --training_conf configs/convnext/convnext_2layer_tune_train_mnist.json --inf_fn_conf configs/convnext/tracin_convnext.json --error mislabelled_uniform --contamination 0.1 --seed 0 --device cpu
	python -m mislabelled_pipeline --data_name fmnist --data_folder data --subset_ratio 0.1 --model_name convnext --model_conf configs/convnext/convnext_2layer_tune_model.json --training_conf configs/convnext/convnext_2layer_tune_train_fmnist.json --inf_fn_conf configs/convnext/tracin_convnext.json --error mislabelled_uniform --contamination 0.1 --seed 0 --device cpu
	python -m mislabelled_pipeline --data_name cifar10 --data_folder data --subset_ratio 0.1 --model_name convnext --model_conf configs/convnext/convnext_2layer_tune_model.json --training_conf configs/convnext/convnext_2layer_tune_train_cifar10.json --inf_fn_conf configs/convnext/tracin_convnext.json --error mislabelled_uniform --contamination 0.1 --seed 0 --device cpu

uniform-class-noise-vit-sigs:
	python -m mislabelled_pipeline --data_name mnist --data_folder data --subset_ratio 0.1 --model_name vit --model_conf configs/vit/vit_2layer_tune_model.json --training_conf configs/vit/vit_2layer_tune_train_mnist.json --inf_fn_conf configs/vit/tracin_vit.json --error mislabelled_uniform --contamination 0.1 --seed 0 --device cpu
	python -m mislabelled_pipeline --data_name fmnist --data_folder data --subset_ratio 0.1 --model_name vit --model_conf configs/vit/vit_2layer_tune_model.json --training_conf configs/vit/vit_2layer_tune_train_fmnist.json --inf_fn_conf configs/vit/tracin_vit.json --error mislabelled_uniform --contamination 0.1 --seed 0 --device cpu
	python -m mislabelled_pipeline --data_name cifar10 --data_folder data --subset_ratio 0.1 --model_name vit --model_conf configs/vit/vit_2layer_tune_model.json --training_conf configs/vit/vit_2layer_tune_train_cifar10.json --inf_fn_conf configs/vit/tracin_vit.json --error mislabelled_uniform --contamination 0.1 --seed 0 --device cpu

# Class-based noise

class-based-noise-resnet:
	python -m mislabelled_pipeline --data_name mnist --data_folder data --subset_ratio 0.1 --model_name resnet20 --model_conf configs/resnet/resnet_2layer_tune_model.json --training_conf configs/resnet/resnet_2layer_tune_train_mnist.json --error mislabelled_cb --contamination 0.1 --seed 0 --device cpu
	python -m mislabelled_pipeline --data_name fmnist --data_folder data --subset_ratio 0.1 --model_name resnet20 --model_conf configs/resnet/resnet_2layer_tune_model.json --training_conf configs/resnet/resnet_2layer_tune_train_fmnist.json --error mislabelled_cb --contamination 0.1 --seed 0 --device cpu
	python -m mislabelled_pipeline --data_name cifar10 --data_folder data --subset_ratio 0.1 --model_name resnet20 --model_conf configs/resnet/resnet_2layer_tune_model.json --training_conf configs/resnet/resnet_2layer_tune_train_cifar10.json --error mislabelled_cb --contamination 0.1 --seed 0 --device cpu

class-based-noise-vit:
	python -m mislabelled_pipeline --data_name mnist --data_folder data --subset_ratio 0.1 --model_name vit --model_conf configs/vit/vit_2layer_tune_model.json --training_conf configs/vit/vit_2layer_tune_train_mnist.json --inf_fn_conf configs/vit/tracin_vit.json --error mislabelled_cb --contamination 0.1 --seed 0 --device cpu
	python -m mislabelled_pipeline --data_name fmnist --data_folder data --subset_ratio 0.1 --model_name vit --model_conf configs/vit/vit_2layer_tune_model.json --training_conf configs/vit/vit_2layer_tune_train_fmnist.json --inf_fn_conf configs/vit/tracin_vit.json --error mislabelled_cb --contamination 0.1 --seed 0 --device cpu
	python -m mislabelled_pipeline --data_name cifar10 --data_folder data --subset_ratio 0.1 --model_name vit --model_conf configs/vit/vit_2layer_tune_model.json --training_conf configs/vit/vit_2layer_tune_train_cifar10.json --inf_fn_conf configs/vit/tracin_vit.json --error mislabelled_cb --contamination 0.1 --seed 0 --device cpu

class-based-noise-convnext:
	python -m mislabelled_pipeline --data_name mnist --data_folder data --subset_ratio 0.1 --model_name convnext --model_conf configs/convnext/convnext_2layer_tune_model.json --training_conf configs/convnext/convnext_2layer_tune_train_mnist.json --inf_fn_conf configs/convnext/tracin_convnext.json --error mislabelled_cb --contamination 0.1 --seed 0 --device cpu
	python -m mislabelled_pipeline --data_name fmnist --data_folder data --subset_ratio 0.1 --model_name convnext --model_conf configs/convnext/convnext_2layer_tune_model.json --training_conf configs/convnext/convnext_2layer_tune_train_fmnist.json --inf_fn_conf configs/convnext/tracin_convnext.json --error mislabelled_cb --contamination 0.1 --seed 0 --device cpu
	python -m mislabelled_pipeline --data_name cifar10 --data_folder data --subset_ratio 0.1 --model_name convnext --model_conf configs/convnext/convnext_2layer_tune_model.json --training_conf configs/convnext/convnext_2layer_tune_train_cifar10.json --inf_fn_conf configs/convnext/tracin_convnext.json --error mislabelled_cb --contamination 0.1 --seed 0 --device cpu


# Anomalies

anomalies-resnet:
	python -m anomaly_pipeline --data_name mnist --data_folder data --subset_ratio 0.1 --model_name resnet20 --model_conf configs/resnet/resnet_2layer_tune_model.json --training_conf configs/resnet/resnet_2layer_tune_train_mnist.json --ood_data_name fmnist --contamination 0.1 --seed 42 --device cpu
	python -m anomaly_pipeline --data_name fmnist --data_folder data --subset_ratio 0.1 --model_name resnet20 --model_conf configs/resnet/resnet_2layer_tune_model.json --training_conf configs/resnet/resnet_2layer_tune_train_fmnist.json --ood_data_name mnist --contamination 0.1 --seed 42 --device cpu
	python -m anomaly_pipeline --data_name cifar10 --data_folder data --subset_ratio 0.1 --model_name resnet20 --model_conf configs/resnet/resnet_2layer_tune_model.json --training_conf configs/resnet/resnet_2layer_tune_train_cifar10.json --ood_data_name fmnist --contamination 0.1 --seed 42 --device cpu

anomalies-vit:
	python -m anomaly_pipeline --data_name mnist --data_folder data --subset_ratio 0.1 --model_name vit --model_conf configs/vit/vit_2layer_tune_model.json --training_conf configs/vit/vit_2layer_tune_train_mnist.json  --inf_fn_conf configs/vit/tracin_vit.json --ood_data_name fmnist --contamination 0.1 --seed 42 --device cpu
	python -m anomaly_pipeline --data_name fmnist --data_folder data --subset_ratio 0.1 --model_name vit --model_conf configs/vit/vit_2layer_tune_model.json --training_conf configs/vit/vit_2layer_tune_train_fmnist.json --inf_fn_conf configs/vit/tracin_vit.json --ood_data_name mnist --contamination 0.1 --seed 42 --device cpu
	python -m anomaly_pipeline --data_name cifar10 --data_folder data --subset_ratio 0.1 --model_name vit --model_conf configs/vit/vit_2layer_tune_model.json --training_conf configs/vit/vit_2layer_tune_train_cifar10.json --inf_fn_conf configs/vit/tracin_vit.json --ood_data_name fmnist --contamination 0.1 --seed 42 --device cpu

anomalies-convnext:
	python -m anomaly_pipeline --data_name mnist --data_folder data --subset_ratio 0.1 --model_name convnext --model_conf configs/convnext/convnext_2layer_tune_model.json --training_conf configs/convnext/convnext_2layer_tune_train_mnist.json  --inf_fn_conf configs/convnext/tracin_convnext.json --ood_data_name fmnist --contamination 0.1 --seed 42 --device cpu
	python -m anomaly_pipeline --data_name fmnist --data_folder data --subset_ratio 0.1 --model_name convnext --model_conf configs/convnext/convnext_2layer_tune_model.json --training_conf configs/convnext/convnext_2layer_tune_train_fmnist.json --inf_fn_conf configs/convnext/tracin_convnext.json --ood_data_name mnist --contamination 0.1 --seed 42 --device cpu
	python -m anomaly_pipeline --data_name cifar10 --data_folder data --subset_ratio 0.1 --model_name convnext --model_conf configs/convnext/convnext_2layer_tune_model.json --training_conf configs/convnext/convnext_2layer_tune_train_cifar10.json --inf_fn_conf configs/convnext/tracin_convnext.json --ood_data_name fmnist --contamination 0.1 --seed 42 --device cpu

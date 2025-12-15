#!/bin/bash
# This script runs run_toy_task.py with different parameter configurations in a grid search

echo "Starting comprehensive grid search experiments..."
i=0

# Configuration toggles - set to 1 to enable, 0 to disable
ENABLE_METHODS=0
ENABLE_ARCHITECTURES=0
ENABLE_NUM_LAYERS=0
ENABLE_NUM_HIDDEN=0
ENABLE_MEMORY=0
ENABLE_ACTIVATION=0
ENABLE_DECODER=0
ENABLE_MIXING=0
ENABLE_NORM_FLAGS=0
ENABLE_ENCODER_FLAG=0
ENABLE_LAYER_OUTPUT_FLAG=0
ENABLE_EXTRA_SKIP_FLAG=0
ENABLE_HAS_NON_LINEARITY_IN_RECURRENCE_FLAG=0
ENABLE_DATASET=0

# Parameter arrays
methods=(ONLINE) 
architectures=(GRU ZUC)
num_layers_arr=(3)
num_hidden_arr=(32)
memory_arr=(3 4)
activations=(sigmoid full_glu)
decoders=(MLP NONE)
mixings=(full none)
norm_flags=("" "--prenorm")
encoder_flags=("" "--encoder")
layer_output_flags=("" "--layer_output")
extra_skip_flags=("" "--extra_skip")
has_non_linearity_in_recurrence_flags=("" "--has_non_linearity_in_recurrence")
datasets=(mooc reddit wikipedia)

# Set default values when disabled
[[ $ENABLE_METHODS -eq 0 ]] && methods=(ONLINE)
[[ $ENABLE_ARCHITECTURES -eq 0 ]] && architectures=(ZUC)
[[ $ENABLE_NUM_LAYERS -eq 0 ]] && num_layers_arr=(4)
[[ $ENABLE_NUM_HIDDEN -eq 0 ]] && num_hidden_arr=(32)
[[ $ENABLE_MEMORY -eq 0 ]] && memory_arr=(2)
[[ $ENABLE_ACTIVATION -eq 0 ]] && activations=(full_glu)
[[ $ENABLE_DECODER -eq 0 ]] && decoders=(MLP)
[[ $ENABLE_MIXING -eq 0 ]] && mixings=(rotational_full)
[[ $ENABLE_NORM_FLAGS -eq 0 ]] && norm_flags=("--prenorm")
[[ $ENABLE_ENCODER_FLAG -eq 0 ]] && encoder_flags=("--encoder")
[[ $ENABLE_LAYER_OUTPUT_FLAG -eq 0 ]] && layer_output_flags=("--layer_output")
[[ $ENABLE_EXTRA_SKIP_FLAG -eq 0 ]] && extra_skip_flags=("--extra_skip")
[[ $ENABLE_HAS_NON_LINEARITY_IN_RECURRENCE_FLAG -eq 0 ]] && has_non_linearity_in_recurrence_flags=("")
[[ $ENABLE_DATASET -eq 0 ]] && datasets=(mooc)

# Grid search - most important parameters in inner loops (change most frequently)
for dataset in "${datasets[@]}"; do
  for has_non_linearity_in_recurrence_flag in "${has_non_linearity_in_recurrence_flags[@]}"; do
    for extra_skip_flag in "${extra_skip_flags[@]}"; do
      for layer_output_flag in "${layer_output_flags[@]}"; do
        for encoder_flag in "${encoder_flags[@]}"; do
          for norm_flag in "${norm_flags[@]}"; do
            for mixing in "${mixings[@]}"; do
              for decoder in "${decoders[@]}"; do
                for activation in "${activations[@]}"; do
                  for memory in "${memory_arr[@]}"; do
                    for num_hidden in "${num_hidden_arr[@]}"; do
                      for num_layers in "${num_layers_arr[@]}"; do
                        for architecture in "${architectures[@]}"; do
                          for method in "${methods[@]}"; do
                            ((i++))

                            echo "Running with method=$method, arch=$architecture, layers=$num_layers, hidden=$num_hidden, memory=$memory, activation=$activation, decoder=$decoder, mixing=$mixing, norm_flag=$norm_flag, encoder_flag=$encoder_flag, layer_output_flag=$layer_output_flag, extra_skip_flag=$extra_skip_flag, has_non_linearity_in_recurrence_flag=$has_non_linearity_in_recurrence_flag, dataset=$dataset"

                            python3 run_toy_task.py \
                              -m $method \
                              -a $architecture \
                              --num_layers $num_layers \
                              --num_hidden $num_hidden \
                              --memory $memory \
                              --activation $activation \
                              --decoder $decoder \
                              --mixing $mixing \
                              $norm_flag \
                              $encoder_flag \
                              $layer_output_flag \
                              $extra_skip_flag \
                              $has_non_linearity_in_recurrence_flag \
                              --dataset $dataset \
                              --task link_classification \
                              --batching_strategy none \
                              --dropout 0.15 \
                              --weight_decay 0.005 \
                              --batch_size 50 \
                              --num_epochs 200 \
                              "--double_dmodel" \
                              --num_gradient_accumulation_steps 32 \
                              --dedupe \
                              --steps_for_scheduler 100 \
                              --lr_schedule cosine \
                              --rec_lr_schedule cosine \
                              --lr_min 1e-6 \

                            echo "Experiment number: $i completed"
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "All grid search experiments completed!"

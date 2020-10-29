#!/bin/bash -x

#wget https://evaluation2-training-scenes.s3.amazonaws.com/interaction-scenes.zip
#wget https://evaluation2-training-scenes.s3.amazonaws.com/intphys-scenes.zip
#
#unzip interaction-scenes.zip
#unzip intphys-scenes.zip -d intphys_scenes
#
#mkdir intphys_scenes/gravity
#mkdir intphys_scenes/object_permanence
#mkdir intphys_scenes/shape_constancy
#mkdir intphys_scenes/spatio_temporal_continuity
#
#mkdir interaction_scenes/retrieval
#mkdir interaction_scenes/traversal
#mkdir interaction_scenes/transferral
#
#mv intphys_scenes/gravity*.json intphys_scenes/gravity
#mv intphys_scenes/object_permanence*.json intphys_scenes/object_permanence
#mv intphys_scenes/shape_constancy*.json intphys_scenes/shape_constancy
#mv intphys_scenes/spatio_temporal_continuity*.json intphys_scenes/spatio_temporal_continuity
#
#mv interaction_scenes/retrieval*.json interaction_scenes/retrieval
#mv interaction_scenes/traversal*.json interaction_scenes/traversal
#mv interaction_scenes/transferral*.json interaction_scenes/transferral
#
#rm interaction-scenes.zip
#rm intphys-scenes.zip

mkdir -p eval3_dataset
wget https://evaluation-training-scenes.s3.amazonaws.com/eval3/training-single-object.zip
wget https://evaluation-training-scenes.s3.amazonaws.com/eval3/training-object-preference.zip

unzip training-single-object.zip
unzip training-object-preference.zip

mv TRAINING_OBJECT_PREFERENCE eval3_dataset
mv TRAINING_SINGLE_OBJECT eval3_dataset

rm training-single-object.zip
rm training-object-preference.zip
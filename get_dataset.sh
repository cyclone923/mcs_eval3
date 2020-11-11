#!/bin/bash -x

wget https://evaluation2-training-scenes.s3.amazonaws.com/interaction-scenes.zip
wget https://evaluation-training-scenes.s3.amazonaws.com/eval3/training-passive-physics.zip

unzip interaction-scenes.zip
unzip training-passive-physics.zip

mkdir evaluation3Training/object_permanence
mkdir evaluation3Training/shape_constancy
mkdir evaluation3Training/spatio_temporal_continuity

mkdir interaction_scenes/retrieval
mkdir interaction_scenes/traversal
mkdir interaction_scenes/transferral

mv evaluation3Training/*ObjectPermanence*.json evaluation3Training/object_permanence
mv evaluation3Training/*ShapeConstancy*.json evaluation3Training/shape_constancy
mv evaluation3Training/*SpatioTemporalContinuity*.json evaluation3Training/spatio_temporal_continuity

mv interaction_scenes/retrieval*.json interaction_scenes/retrieval
mv interaction_scenes/traversal*.json interaction_scenes/traversal
mv interaction_scenes/transferral*.json interaction_scenes/transferral

rm interaction-scenes.zip
rm training-passive-physics.zip

mkdir -p eval3_dataset
wget https://evaluation-training-scenes.s3.amazonaws.com/eval3/training-single-object.zip
wget https://evaluation-training-scenes.s3.amazonaws.com/eval3/training-object-preference.zip

unzip training-single-object.zip
unzip training-object-preference.zip

mv TRAINING_OBJECT_PREFERENCE eval3_dataset
mv TRAINING_SINGLE_OBJECT eval3_dataset

rm training-single-object.zip
rm training-object-preference.zip
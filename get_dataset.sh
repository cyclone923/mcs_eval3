#!/bin/bash -x

# skip if already downloaded
if [ -d "interaction_scenes/retrieval" ]; then
  exit 0
fi

wget https://evaluation2-training-scenes.s3.amazonaws.com/interaction-scenes.zip

unzip interaction-scenes.zip
unzip training-passive-physics.zip

mkdir -p interaction_scenes/retrieval
mkdir -p interaction_scenes/traversal
mkdir -p interaction_scenes/transferral

mv interaction_scenes/retrieval*.json interaction_scenes/retrieval
mv interaction_scenes/traversal*.json interaction_scenes/traversal
mv interaction_scenes/transferral*.json interaction_scenes/transferral

rm interaction-scenes.zip

#VOE

wget https://evaluation-training-scenes.s3.amazonaws.com/eval3/training-passive-physics.zip

unzip training-passive-physics.zip

mkdir -p evaluation3Training/object_permanence
mkdir -p evaluation3Training/shape_constancy
mkdir -p evaluation3Training/spatio_temporal_continuity

mv evaluation3Training/*ObjectPermanence*.json evaluation3Training/object_permanence
mv evaluation3Training/*ShapeConstancy*.json evaluation3Training/shape_constancy
mv evaluation3Training/*SpatioTemporalContinuity*.json evaluation3Training/spatio_temporal_continuity

rm training-passive-physics.zip

#Agent VOE

mkdir -p eval3_dataset
wget https://evaluation-training-scenes.s3.amazonaws.com/eval3/training-single-object.zip
wget https://evaluation-training-scenes.s3.amazonaws.com/eval3/training-object-preference.zip

unzip training-single-object.zip
unzip training-object-preference.zip

mv TRAINING_OBJECT_PREFERENCE eval3_dataset
mv TRAINING_SINGLE_OBJECT eval3_dataset

rm training-single-object.zip
rm training-object-preference.zip

mkdir -p different_scenes
cp eval3_dataset/TRAINING_OBJECT_PREFERENCE/preference_0001_01.json different_scenes
cp interaction_scenes/transferral/transferral_goal-0001.json different_scenes
cp evaluation3Training/object_permanence/eval3TrainingObjectPermanence_0001_01.json different_scenes


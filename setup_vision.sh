# if not installed already
if ! [ pip list | grep DCNv2 ]; then
  cd vision/instSeg/external/DCNv2
  python setup.py build develop
  cd ../../

  wget https://oregonstate.box.com/shared/static/0syjouwkkpm0g1zbnt1riheshfvtd2by.pth
  mv 0syjouwkkpm0g1zbnt1riheshfvtd2by.pth dvis_resnet50_mc.pth

  cd ../../visionmodule
  wget https://oregonstate.box.com/shared/static/zmvcjyumltkziqfqbcqodkh6dgecikci.pth
  mv zmvcjyumltkziqfqbcqodkh6dgecikci.pth dvis_resnet50_mc_voe.pth 

  cd ../
fi

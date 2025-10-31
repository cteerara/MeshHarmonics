#!/bin/bash

src_mesh=la_003_cs.stl
tgt_mesh=la_003.stl
num_modes=36
# out_mesh=mesh_T.stl
num_steps=40

mkdir -p steps
cp $src_mesh steps/tmp_mesh_0.stl
python3 deform_sphere.py $src_mesh $tgt_mesh $num_modes steps/tmp_mesh_1.stl

for (( i=1 ; i<$((num_steps+1)) ; i++ ))
do
    src_mesh=tmp_mesh_${i}.stl
    j=$((i+1))
    out_mesh=tmp_mesh_${j}.stl
    python3 deform_sphere.py steps/$src_mesh $tgt_mesh $num_modes steps/$out_mesh
done

python3 stl_to_pvd.py steps/ tmp_mesh_ $num_steps

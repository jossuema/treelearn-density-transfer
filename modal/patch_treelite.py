"""Three tweaks to the TreeLite3D tester. They are applied when building the image.

1) The output files are named with the loop index (instance_pred_0.npy).
   The loop order comes from glob.glob, which is not sorted, so that index does not
   identify the plot. It is switched to use the file name.
2) Saving origin_coord is commented out in the original. Without it there is no way to
   check that the labels come back in the input order. It is uncommented.
3) pred_label comes out of np.argmax over the proposal masks. A point that falls in no
   proposal has its whole row at zero and argmax returns 0, so it ends up labelled as
   instance 0. With no way to know which ones those are, instance 0 swallows the ground
   and the background of the plot. Which points are assigned is saved separately.

Nothing in the model or in the computation is touched. Each substitution is verified to match exactly once before it is applied.
"""
F = "/workspace/TreeLite3D/pointcept/engines/test.py"
s = open(F).read()

cambios = [
 ('            pred_save_path_inst = os.path.join(save_path, "instance_pred_%s.npy" %(str(i)))',
  '            name = self.test_loader.dataset.get_data_name(i)\n'
  '            pred_save_path_inst = os.path.join(save_path, "instance_pred_%s.npy" %(name))'),
 ('            pred_save_path_sem = os.path.join(save_path, "semantic_pred_%s.npy" %(str(i)))',
  '            pred_save_path_sem = os.path.join(save_path, "semantic_pred_%s.npy" %(name))'),
 ('            pred_save_path_orig_coord = os.path.join(save_path, "orig_coord_%s.npy" %(str(i)))',
  '            pred_save_path_orig_coord = os.path.join(save_path, "orig_coord_%s.npy" %(name))'),
 ('            # np.save(pred_save_path_orig_coord, input_dict["origin_coord"].cpu().numpy())',
  '            np.save(pred_save_path_orig_coord, input_dict["origin_coord"].cpu().numpy())'),
 ('            pred_save_path_shifted_coord = os.path.join(save_path, "shifted_coord_%s.npy" %(str(i)))',
  '            np.save(os.path.join(save_path, "assigned_%s.npy" %(name)),\n'
  '                    (output_dict["pred_masks"].sum(0) > 0).cpu().numpy())\n'
  '            pred_save_path_shifted_coord = os.path.join(save_path, "shifted_coord_%s.npy" %(name))'),
]
for old, new in cambios:
    if s.count(old) != 1:
        raise SystemExit(f"patch failed, {s.count(old)} matches of:\n{old}")
    s = s.replace(old, new)
open(F, "w").write(s)
print("patch applied, 5 substitutions")

"""Dos retoques al probador de TreeLite3D. Se aplican al construir la imagen.

1) Los ficheros de salida se nombran con el indice del bucle (instance_pred_0.npy).
   El orden del bucle viene de glob.glob, que no esta ordenado, asi que ese indice no
   identifica la parcela. Se pasa a usar el nombre del fichero.
2) El guardado de origin_coord esta comentado en el original. Sin el no hay forma de
   comprobar que las etiquetas vuelven en el orden de entrada. Se descomenta.
3) pred_label sale de np.argmax sobre las mascaras de propuesta. Un punto que no cae en
   ninguna propuesta tiene la fila entera a cero y argmax devuelve 0, asi que acaba
   etiquetado como la instancia 0. Sin saber cuales son, la instancia 0 se traga el
   suelo y el fondo de la parcela. Se guarda aparte que puntos si estan asignados.

No se toca nada del modelo ni del calculo. Se verifica que cada sustitucion se aplique.
"""
F = "/workspace/TreeLite3D/pointcept/engines/test.py"
s = open(F).read()

cambios = [
 ('            pred_save_path_inst = os.path.join(save_path, "instance_pred_%s.npy" %(str(i)))',
  '            nombre = self.test_loader.dataset.get_data_name(i)\n'
  '            pred_save_path_inst = os.path.join(save_path, "instance_pred_%s.npy" %(nombre))'),
 ('            pred_save_path_sem = os.path.join(save_path, "semantic_pred_%s.npy" %(str(i)))',
  '            pred_save_path_sem = os.path.join(save_path, "semantic_pred_%s.npy" %(nombre))'),
 ('            pred_save_path_orig_coord = os.path.join(save_path, "orig_coord_%s.npy" %(str(i)))',
  '            pred_save_path_orig_coord = os.path.join(save_path, "orig_coord_%s.npy" %(nombre))'),
 ('            # np.save(pred_save_path_orig_coord, input_dict["origin_coord"].cpu().numpy())',
  '            np.save(pred_save_path_orig_coord, input_dict["origin_coord"].cpu().numpy())'),
 ('            pred_save_path_shifted_coord = os.path.join(save_path, "shifted_coord_%s.npy" %(str(i)))',
  '            np.save(os.path.join(save_path, "asignado_%s.npy" %(nombre)),\n'
  '                    (output_dict["pred_masks"].sum(0) > 0).cpu().numpy())\n'
  '            pred_save_path_shifted_coord = os.path.join(save_path, "shifted_coord_%s.npy" %(nombre))'),
]
for viejo, nuevo in cambios:
    if s.count(viejo) != 1:
        raise SystemExit(f"parche fallido, {s.count(viejo)} coincidencias de:\n{viejo}")
    s = s.replace(viejo, nuevo)
open(F, "w").write(s)
print("parche aplicado, 5 sustituciones")

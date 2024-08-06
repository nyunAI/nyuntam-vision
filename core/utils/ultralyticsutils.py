import shutil
import os
from pycocotools.coco import COCO


def create_ultralytics_folder_type(
    root, coco_dest_folder="coco_format_data", ann_folder_name="instances_val2017.json"
):
    # used to make sure not trailing / are there
    if root.endswith("/"):
        root = root[:-1]
    os.makedirs(
        os.path.join(root, coco_dest_folder, "images", "train2017"), exist_ok=True
    )
    os.makedirs(
        os.path.join(root, coco_dest_folder, "labels", "train2017"), exist_ok=True
    )
    os.makedirs(
        os.path.join(root, coco_dest_folder, "labels", "val2017"), exist_ok=True
    )
    os.makedirs(
        os.path.join(root, coco_dest_folder, "images", "val2017"), exist_ok=True
    )

    train_images_folder = os.path.join(root, "train2017")
    val_images_folder = os.path.join(root, "val2017")
    train_ann_path = os.path.join(root, "annotations", ann_folder_name)
    val_ann_path = os.path.join(root, "annotations", ann_folder_name)

    copy_images(
        train_images_folder, os.path.join(root, coco_dest_folder, "images", "train2017")
    )
    copy_images(
        val_images_folder, os.path.join(root, coco_dest_folder, "images", "val2017")
    )
    make_labels(root, train_ann_path, "train2017", coco_dest_folder)
    make_labels(root, val_ann_path, "val2017", coco_dest_folder)


def copy_images(src, dst):
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return


def make_labels(root, ann_file, folname, coco_dest_folder):
    def get_image_name(coco, id):
        image_info = coco.loadImgs(id)[0]
        image_name = image_info["file_name"]
        return image_name

    def normalize_bb(coco, id, bb):
        image_info = coco.loadImgs(id)[0]
        img_width = image_info["width"]
        img_height = image_info["height"]
        normalized_bbox = [
            bbox[0] / img_width,
            bbox[1] / img_height,
            bbox[2] / img_width,
            bbox[3] / img_height,
        ]
        return normalized_bbox

    def get_class_id(c_id):
        name = coco.loadCats([category_id])[0]["name"]
        return name_dict[name]

    coco_annotations_file = ann_file
    coco = COCO(coco_annotations_file)
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]
    name_dict = {name: i for i, name in enumerate(cat_names)}
    id_boxes = {}
    for annotation in coco.dataset["annotations"]:
        image_id = annotation["image_id"]
        bbox = annotation["bbox"]
        bbox = normalize_bb(coco, image_id, bbox)
        category_id = annotation["category_id"]

        c_id = coco.loadCats([category_id])[0]["id"]
        c = [get_class_id(c_id)]
        c.extend(bbox)
        if image_id in id_boxes:
            id_boxes[image_id].append(c)
        else:
            id_boxes[image_id] = [c]

    for k in id_boxes.keys():
        image_name = get_image_name(coco, k)
        file_name = f"{image_name.split('.')[0]}.txt"
        os.makedirs(f"{root}/{coco_dest_folder}/labels/{folname}", exist_ok=True)

        with open(
            os.path.join(f"{root}/{coco_dest_folder}/labels/{folname}/{file_name}"), "w"
        ) as file:
            box = id_boxes[k]
            string_box = []
            for bb in box:
                string_box.append(" ".join([str(i) for i in bb]) + "\n")
            file.writelines(string_box)

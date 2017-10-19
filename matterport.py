class MPDataset(COCODataset):

    def load_ade20k(self, dataset_dir, subset, class_map):
        """Load the specified set of data from the ADE20K dataset and add it to 
        the data already in this object.

        subset:
            "train": all training set
            "val": validation set
            "train_top" or "val_top": Classes with a lot of examples.
            "train_pick" or "val_pick": Selected classes
        """
        assert subset in ["train", "val", "train_top", "val_top", "train_pick", "val_pick"]
        dataset_dir = os.path.join(dataset_dir, "train" if subset.startswith("train") else "val")
        
        # Load matlab index file
        matlab = scipy.io.loadmat(os.path.join(dataset_dir, "index_ade20k.mat"))
        index = {}
        index['object_name'] = [f[0] for f in matlab['index']["objectnames"][0][0][0]]
        index['object_count'] = matlab['index']["objectcounts"][0][0][:,0]
        index['object_presence'] = matlab['index']['objectPresence'][0][0]
        index['image_path'] = [os.path.join(d[0][18:], f[0])
                               for d, f in zip(matlab["index"]["folder"][0][0][0], 
                                               matlab["index"]["filename"][0][0][0])]
        index['is_part'] = matlab['index']['proportionClassIsPart'][0][0][:,0]
        index['train_ids'] = [i for i, s in enumerate(index['image_path']) if "/training/" in s]
        index['val_ids'] = [i for i, s in enumerate(index['image_path']) if "/validation/" in s]
        del matlab  # free memory
        
        # Select image IDs (training or validation).
        ade_image_ids = np.array(index['train_ids'] if subset.startswith("train") else index['val_ids'])
        
        if subset == "top":
            # List top level objects with many examples that are not part of other objects.
            ade_class_ids = [i for i, c in enumerate(index['object_count']) 
                             if c * (1- index['is_part'][i]) >= 1500]
            # Exclude some classes TODO: consider removing these exceptions
            ade_class_ids.remove(2977)  # wall
            ade_class_ids.remove(446)   # ceiling
            ade_class_ids.remove(975)   # floor
            ade_class_ids.remove(2419)  # sky
            ade_class_ids.remove(2854)  # tree
            ade_class_ids.remove(311)   # building
            ade_class_ids.remove(2130)  # road
            ade_class_ids.remove(2376)  # sidewalk
            ade_class_ids.remove(837)   # ground
            ade_class_ids.remove(1609)  # mountain
            ade_class_ids.remove(400)   # car
            
            # Filter images by the selected classes
            ade_image_ids = ade_image_ids[np.where(np.sum(
                index['object_presence'][ade_class_ids][:, ade_image_ids], axis=0) > 0)[0]]
        elif subset == "pick":
            ade_class_ids = [mapping["ade"] for name, mapping in class_map if "ade" in mapping]
            
            # Filter images by the selected classes
            # This is approximate because ADE can have a class as a part of
            # another object, which we don't use. This is handled by skipping
            # the image if we load it and find that it has no mask we can use.
            ade_image_ids = ade_image_ids[np.where(np.sum(
                index['object_presence'][ade_class_ids][:, ade_image_ids], axis=0) > 0)[0]]
        elif subset == "all":
            # All class IDs
            ade_class_ids = np.arange(len(index['object_count']))
        
        # Arrange data and add to current dataset
        if not class_map:
            class_info = [{"map": ("ade", e),
                           "name": index['object_name'][e]} 
                          for e in ade_class_ids]
        else:
            class_info = []
        image_info = [{"id": e, 
                       "ds": "ade",
                       "path": os.path.join(dataset_dir, index['image_path'][e])} 
                      for e in ade_image_ids]
        self.append_data(class_info, image_info)
        
    def load_mirrors(self, dataset_dir, subset, class_map):
        """Loads the mirrors dataset, which is described in 
        the explore_mirror_dataset.ipynb notebook.
        """
        # This dataset has one class only: mirror.
        if not class_map:
            class_info = [{"map": [("mirrors", 1)],
                           "name": "mirror"}]
        else:
            class_info = []

        # Data location
        images_dir = os.path.join(dataset_dir, 'rawData')
        masks_dir = os.path.join(dataset_dir, 'cleanedMasks')
        meta_path = os.path.join(dataset_dir, 'metaMirror.txt')

        # Read metadata file
        with open(meta_path) as f:
            metadata = list(json.load(f).items())
            
        # Training/validation split.
        # For consistent splitting, sort the metadata by the file name and
        # take the first 90% as training and the rest as validation.
        metadata.sort(key=lambda m: m[0])
        split = int(len(metadata) * 0.9)
        metadata = metadata[:split] if subset == "train" else metadata[split:]
        
        # Loop through image details
        image_info = []
        for info in metadata:
            # Extrac directory and image file name from the first field of info.
            file_name = info[0]
            m = re.fullmatch(r"(.*)/(.*)\_(.+)\_(\w+)\.png", file_name)
            model_id = m.group(1)
            pano_id = m.group(2)
            skybox = m.group(3)  # values are skybox0, 1, 2, 3, 4, 5
            content = m.group(4)  # simple=RGB, depth=depth image, wall=uncleaned mirror mask

            # Image and mask paths
            image_path = os.path.join(images_dir, model_id, 
                                      "{}_{}_simple.png".format(pano_id, skybox, content))
            mask_path = os.path.join(masks_dir, "{}_{}_{}_wall.png".format(model_id, pano_id, skybox))

            image_info.append({
                "id": "/".join([model_id, pano_id, skybox]),
                "ds": "mirrors",
                "path": image_path,
                "mask_path": mask_path,
                "bbox": [(b[1][0], b[1][1], b[1][2], b[1][3]) 
                         for b in info[1]],  # y1, x1, y2, x2 in image coordinates
            })
            
        self.append_data(class_info, image_info)

    def load_matterport(self, root_dir, subset, class_map=None):
        if subset == "train":
            # Use Matterport class map if none if provided to overwrite it.
            class_map = class_map or self.get_matterport_class_map()
            self.setup_class_map(class_map)
            # Load data from COCO, ADE20K, and the mirrors datasets.
            # COCO train
            self.load_coco(os.path.join(root_dir, "datasets/coco2014"), subset, class_map)
            # COCO val35k
            self.load_coco(os.path.join(root_dir, "datasets/coco2014"), "val35k", class_map)
            # ADE20K
            dataset_dir = os.path.join(root_dir, "datasets/ADE20K_2016_07_26")
            self.load_from_ade20k("train_pick", class_map)
            # Mirrors
            # todo: self.load_mirrors(os.path.join(root_dir, 'datasets/mirrorDetection/'), "train", class_map)
        elif subset == "val":
            # Use Matterport class map if none if provided to overwrite it.
            class_map = class_map or self.get_matterport_class_map()
            self.setup_class_map(class_map)
            # Load select classes from COCO, ADE20K, and the mirrors datasets.
            # COCO
            dataset_dir = os.path.join(root_dir, "datasets/coco2014")
            coco = COCO(os.path.join(dataset_dir, "annotations/instances_minival2014.json"))
            self.load_coco(os.path.join(root_dir, "datasets/coco2014"), "val", class_map)
            # ADE
            self.load_ade20k(os.path.join(root_dir, "datasets/ADE20K_2016_07_26"), "val_pick", class_map)
            # Mirrors
            # todo: self.load_mirrors(os.path.join(root_dir, 'datasets/mirrorDetection/'), "val", class_map)

            
    def load_mask(self, image_id):
        """Load instance masks for the given image.
        
        Different datasets use different ways to store masks. This 
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        instance_masks = []
        class_ids = []
        
        # Which dataset?
        if image_info["ds"] == "ade":
            # Load mask. Path is similar to image path but with _seg.png suffix.
            path = image_info["path"].replace(".jpg", "_seg.png")
            ade_mask = skimage.io.imread(path)
            # ADE masks use R & G channels to encode class IDs and B to encode instance IDs.
            # Merge all into one to get a unique number per instance
            # Convert to uint32
            semantic_mask = (ade_mask[:,:,0]/10)*256 + ade_mask[:,:,1]
            semantic_mask = semantic_mask.astype(np.uint32)*256 + ade_mask[:,:,2]
            
            # Get unique instance ids (excluding 0)
            instance_ids = np.delete(np.unique(semantic_mask), 0)
            # Reconstruct the mask such that it's a [height, width, num_instances] tensor
            for instance_id in instance_ids:
                # ADE Class ID. In images, ADE classes are 1 based, so subtrace 1.
                ade_class_id = instance_id // 256 - 1
                # Map to internal class ID. None indicates that we don't care for this class
                class_id = self.external_to_class_id.get("ade"+str(ade_class_id))
                if class_id:
                    class_ids.append(class_id)
                    instance_masks.append(np.where(semantic_mask == instance_id, True, False))
        
        elif image_info["ds"] == "mirrors":
            # Load mask and convert to boolean array
            mask = skimage.io.imread(image_info["mask_path"])
            assert set(np.unique(mask).tolist()).issubset(set([0, 255]))
            mask = mask.astype(np.bool)
            
            # All instances are included in one mask, so the bounding
            # boxes help separate the instances.
            for bbox in image_info["bbox"]:
                # mask
                y1, x1, y2, x2 = bbox
                m = np.zeros(mask.shape, dtype=np.bool)
                m[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
                # TODO: There is a bug in the mirrors dataset where some
                #       bounding boxes encapsulate empty areas. Skip those.
                if np.sum(m) == 0:
                    continue
                # class
                class_id = self.external_to_class_id.get("mirrors1") # only one class in this set
                # Add to masks list
                instance_masks.append(m)
                class_ids.append(class_id)
        else:
            return super(self.__class__).load_mask(image_id)
                
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
        else:
            mask = np.empty([0, 0, 0])
            class_ids = np.empty([0], np.int32)
        return mask, class_ids
                        
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["ds"] == "ade":
            images_dir = os.path.join(self.root_dir, "datasets/ADE20K_2016_07_26/images") # xxx
            path = info['path'].replace(images_dir, "")
            # Replace the last / with # because ADE website can't take
            # the image name in the url
            path = path[:path.rindex("/")] + "/#" + path[path.rindex("/")+1:]
            return "http://groups.csail.mit.edu/vision/datasets/ADE20K/browse.php/?dirname="                     + path
        elif info["ds"] == "mirrors":
            return info["path"]
        else:
            super(self.__class__).image_reference(self, image_id)
            
            
    def get_matterport_class_map(self):
        """Maps classes from different datasets to allow building a unified set with
        consistent classes.
        """
        class_map = [
        #     ("person",       "coco.1", "ade.1830"),
        #     ("bottle",       "coco.?", "ade.248"),
            ("chair",        "coco.62", "ade.470"),
            ("couch",        "coco.63", "ade.2472"),
            ("bed",          "coco.65", "ade.164"),
            ("table",        "coco.67", "ade.2683"),  # coco=dining table, ade=table
            ("vase",         "coco.86", "ade.2931"),
        #     ("book",         "coco.84", "ade.235"),
            ("toilet",       "coco.70"),
            ("tv",           "coco.72"),
            ("laptop",       "coco.73"),
            ("microwave",    "coco.78"),
            ("oven",         "coco.79"),
            ("toaster",      "coco.80"),
            ("sink",         "coco.81"),
            ("refrigerator", "coco.82"),
        #     ("clock",        "coco.85"),
            ("box",          "ade.265"),
            ("cabinet",      "ade.349"),
        #     ("pillar",       "ade.580"),
            ("curtain",      "ade.686"),
            ("cushion",      "ade.688"),
            ("door",         "ade.773"),
        #     ("fence",        "ade.906"),
            ("lamp",         "ade.1394"),
            ("light",        "ade.1450"),
            ("mirror",       "ade.1563", "mirrors.1"),
            ("painting",     "ade.1734"),
            ("pillow",       "ade.1868"),
            ("flowerpot",    "ade.1980"),
            ("rug",          "ade.2177"),
        #     ("seat",         "ade.2271"),
            ("shelf",        "ade.2328"),
            ("spotlight",    "ade.2508"),
            ("wall socket",  "ade.2981"),
            ("window",       "ade.3054"),
        ]
        return class_map


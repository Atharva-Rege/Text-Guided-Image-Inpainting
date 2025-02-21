class BirdsDataset(Dataset):

    def __init__(self, root, split = "train", transformation = None, captions_per_img = 10, bbox_path= None, WORDS_NUM= 20):
        super().__init__()

        self.root= root
        self.split = split
        self.split_dir = os.path.join(root, split)
        self.WORDS_NUM= WORDS_NUM
        self.transform = transformation
        
        self.embeddings_num = captions_per_img

        self.train_filenames= self.get_filenames(self, os.path.join(root, "train"))
        self.test_filenames = self.get_filenames(self, os.path.join(root, "test"))

        self.bboxes = {}
        if bbox_path is not None:
            with open(bbox_path, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    sno = parts[0]
                    fname = parts[1].split("/")[-1]  # Extract filename only
                    coords = list(map(float, parts[2:]))  # Convert x1, x2, y1, y2 to floats
                    self.bboxes[fname] = coords

        self.captions, self.ixtoword, self.wordtoix, self.n_words = self.get_text_data(root, split)

        self.class_to_id = {folder: idx for idx, folder in enumerate(sorted(os.listdir(self.split_dir)))}


    def get_filenames(self, root, split_dir):

        file_names=[]

        for class_folder in sorted(os.listdir(split_dir)):

            for img_name in sorted(os.listdir(os.path.join(split_dir, class_folder))):

                if not img_name.endswith('_rgb.jpg'):
                    file_names.append(os.path.join(class_folder, img_name))

        return file_names
    
    def load_captions(self, root, file_names):
        
        text_dir = os.path.join(root, "text_c10")
        all_captions=[]

        for filename in file_names:
            caps_path = os.path.join(text_dir, filename[:-3]+"txt")

            with open(caps_path, "r") as f:
                captions = f.read().encode('utf-8').decode('utf8').split('\n')

                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)

                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filename, cnt))
                    
        return all_captions

    def build_dict(self, train_captions, test_captions):

        word_counts = defaultdict(float)

        captions_list = train_captions + test_captions

        for caption in captions_list:
            for word in caption:
                word_counts[word]+=1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ### 1 based indexing over vocab-list, 0th index is taken by EOS
        ixtoword = {}
        ixtoword[0] = '<end>'

        wordtoix = {}
        wordtoix['<end>'] = 0

        ix = 1

        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1
 
        ###converting ["A", "B", ..] to [1,2,....] using wordtoix, encoding the tokens into numerals
        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)


        ### IN THE ORIGINAL CODE THEY ALSO RETURN TEST_CAPTIONS_TOGETHER
        ### I THINK IT MAKES SENSE, BECAUSE ONLY THEN WOULD THE TRAIN AND TEST VOCABS MATCH
        return [train_captions_new, test_captions_new, ixtoword, wordtoix, len(ixtoword)]

    def get_text_data(self, root, split):

        train_captions = self.load_captions(root, self.train_filenames)
        test_captions = self.load_captions(root, self.test_filenames)

        [train_captions, test_captions, ixtoword, wordtoix, n_words] = self.build_dict(train_captions, test_captions)

        if( split == "train"):
            return  train_captions, ixtoword, wordtoix, n_words
        else:
            return  test_captions, ixtoword, wordtoix, n_words

    def get_img(self, img_path, transform=None, bbox=None):

        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)
            img = img.crop([x1, y1, x2, y2])

        if transform is not None:
            img = transform(img)
        return img  ### IN THE ORIGINAL CODE THEY RETURN AS A LIST BECAUSE THEY'RE USING BRANCHES (BASICALLY DIFFERENT SIZES OF THE SAME IMG) 
    
    def get_caption(self, sent_idx):

        # get caption corresponding to sent_idx
        sent_caption = np.asarray(self.captions[sent_idx]).astype('int64')

        ## If the caption already contains an EOS (0) token
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)

        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>') 
        x = np.zeros((self.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= self.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = self.WORDS_NUM

        # Convert indices back to words
        caption_words = [self.ixtoword[idx] for idx in sent_caption if idx in self.ixtoword]
        caption_text = " ".join(caption_words)        
        
        return x, x_len, caption_text

    def __len__(self):
        if(self.split == "train"):
            return len(self.train_filenames)
        
        return len(self.test_filenames)

    def __getitem__(self, index):
        
        if self.split=="train":
            filename= self.train_filenames[index]
            filepath = os.path.join(self.split_dir,filename)
        else:
            filename = self.test_filenames[index]
            filepath = os.path.join(self.split_dir,filename)

        bbox= self.bboxes[filename.split("/")[-1]]

        image = self.get_img(filepath, self.transform,bbox)

        sentence_idx= random.randint(0, self.embeddings_num)
        new_idx = index * self.embeddings_num + sentence_idx

        captions, len_caps, caption_text = self.get_caption(new_idx)

        class_folder = filename.split(os.sep)[0]  # Extract the class folder name
        # print(class_folder, filename )
        class_id = self.class_to_id[class_folder]  # Map class folder to class ID

        return image, captions, len_caps, class_id, caption_text   

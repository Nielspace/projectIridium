class artengine(nn.Module):
    def __init__(
        self,
        *,
        text=None,
        img=None,
        encoding=None,
        text_min = "",
        lr = .07,
        image_size = 512,
        gradient_accumulate_every = 1,
        save_every = 50,
        epochs = 20,
        iterations = 1050,
        save_progress = False,
        bilinear = False,
        open_folder = True,
        seed = None,
        append_seed = False,
        torch_deterministic = False,
        max_classes = None,
        class_temperature = 2.,
        save_date_time = False,
        save_best = False,
        experimental_resample = False,
        ema_decay = 0.99,
        num_cutouts = 128,
        center_bias = False,
    ):
        super().__init__()

        if torch_deterministic:
            assert not bilinear, 'the deterministic (seeded) operation does not work with interpolation (PyTorch 1.7.1)'
            torch.set_deterministic(True)

        self.seed = seed
        self.append_seed = append_seed

        if exists(seed):
            print(f'setting seed of {seed}')
            if seed == 0:
                print('you can override this with --seed argument in the command line, or --random for a randomly chosen one')
            torch.manual_seed(seed)

        self.epochs = epochs
        self.iterations = iterations

        model = BigSleep(
            image_size = image_size,
            bilinear = bilinear,
            max_classes = max_classes,
            class_temperature = class_temperature,
            experimental_resample = experimental_resample,
            ema_decay = ema_decay,
            num_cutouts = num_cutouts,
            center_bias = center_bias,
        ).cuda()

        self.model = model

        self.lr = lr
        self.optimizer = Adam(model.model.latents.model.parameters(), lr)
        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_every = save_every

        self.save_progress = save_progress
        self.save_date_time = save_date_time

        self.save_best = save_best
        self.current_best_score = 0

        self.open_folder = open_folder
        self.total_image_updates = (self.epochs * self.iterations) / self.save_every
        self.encoded_texts = {
            "max": [],
            "min": []
        }
        # create img transform
        self.clip_transform = create_clip_img_transform(224)
        # create starting encoding
        self.set_clip_encoding(text=text, img=img, encoding=encoding, text_min=text_min)
    
    @property
    def seed_suffix(self):
        return f'.{self.seed}' if self.append_seed and exists(self.seed) else ''

    def set_text(self, text):
        self.set_clip_encoding(text = text)

    def create_clip_encoding(self, text=None, img=None, encoding=None):
        self.text = text
        self.img = img
        if encoding is not None:
            encoding = encoding.cuda()
        #elif self.create_story:
        #    encoding = self.update_story_encoding(epoch=0, iteration=1)
        elif text is not None and img is not None:
            encoding = (self.create_text_encoding(text) + self.create_img_encoding(img)) / 2
        elif text is not None:
            encoding = self.create_text_encoding(text)
        elif img is not None:
            encoding = self.create_img_encoding(img)
        return encoding

    def create_text_encoding(self, text):
        tokenized_text = tokenize(text).cuda()
        with torch.no_grad():
            text_encoding = perceptor.encode_text(tokenized_text).detach()
        return text_encoding
    
    def create_img_encoding(self, img):
        if isinstance(img, str):
            img = Image.open(img)
        normed_img = self.clip_transform(img).unsqueeze(0).cuda()
        with torch.no_grad():
            img_encoding = perceptor.encode_image(normed_img).detach()
        return img_encoding
    
    
    def encode_multiple_phrases(self, text, img=None, encoding=None, text_type="max"):
        if text is not None and "|" in text:
            self.encoded_texts[text_type] = [self.create_clip_encoding(text=prompt_min, img=img, encoding=encoding) for prompt_min in text.split("|")]
        else:
            self.encoded_texts[text_type] = [self.create_clip_encoding(text=text, img=img, encoding=encoding)]

    def encode_max_and_min(self, text, img=None, encoding=None, text_min=""):
        self.encode_multiple_phrases(text, img=img, encoding=encoding)
        if text_min is not None and text_min != "":
            self.encode_multiple_phrases(text_min, img=img, encoding=encoding, text_type="min")

    def set_clip_encoding(self, text=None, img=None, encoding=None, text_min=""):
        self.current_best_score = 0
        self.text = text
        self.text_min = text_min
        
        if len(text_min) > 0:
            text = text + "_wout_" + text_min[:255] if text is not None else "wout_" + text_min[:255]
        text_path = create_text_path(text=text, img=img, encoding=encoding)
        if self.save_date_time:
            text_path = datetime.now().strftime("%y%m%d-%H%M%S-") + text_path

        self.text_path = text_path
        self.filename = Path(f'./{text_path}{self.seed_suffix}.png')
        self.encode_max_and_min(text, img=img, encoding=encoding, text_min=text_min) # Tokenize and encode each prompt

    # def reset(self):
    #     self.model.reset()
    #     self.model = self.model.cuda()
    #     self.optimizer = Adam(self.model.model.latents.parameters(), self.lr)

    # def train_step(self, epoch, i, pbar=None):
    #     total_loss = 0

    #     for _ in range(self.gradient_accumulate_every):
    #         out, losses = self.model(self.encoded_texts["max"], self.encoded_texts["min"])
    #         loss = sum(losses) / self.gradient_accumulate_every
    #         total_loss += loss
    #         loss.backward()

    #     self.optimizer.step()
    #     self.model.model.latents.update()
    #     self.optimizer.zero_grad()

    #     if (i + 1) % self.save_every == 0:
    #         with torch.no_grad():
    #             self.model.model.latents.eval()
    #             out, losses = self.model(self.encoded_texts["max"], self.encoded_texts["min"])
    #             top_score, best = torch.topk(losses[2], k=1, largest=False)
    #             image = self.model.model()[best].cpu()
    #             self.model.model.latents.train()

    #             save_image(image, str(self.filename))
    #             if pbar is not None:
    #                 pbar.update(1)
    #             else:
    #                 print(f'image updated at "./{str(self.filename)}"')

    #             if self.save_progress:
    #                 total_iterations = epoch * self.iterations + i
    #                 num = total_iterations // self.save_every
    #                 save_image(image, Path(f'./{self.text_path}.{num}{self.seed_suffix}.png'))

    #             if self.save_best and top_score.item() < self.current_best_score:
    #                 self.current_best_score = top_score.item()
    #                 save_image(image, Path(f'./{self.text_path}{self.seed_suffix}.best.png'))

    #     return out, total_loss

    # def forward(self):
    #     penalizing = ""
    #     if len(self.text_min) > 0:
    #         penalizing = f'penalizing "{self.text_min}"'
    #     print(f'Imagining "{self.text_path}" {penalizing}...')
        
    #     with torch.no_grad():
    #         self.model(self.encoded_texts["max"][0]) # one warmup step due to issue with CLIP and CUDA

    #     if self.open_folder:
    #         open_folder('./')
    #         self.open_folder = False

    #     image_pbar = tqdm(total=self.total_image_updates, desc='image update', position=2, leave=True)
    #     for epoch in trange(self.epochs, desc = '      epochs', position=0, leave=True):
    #         pbar = trange(self.iterations, desc='   iteration', position=1, leave=True)
    #         image_pbar.update(0)
    #         for i in pbar:
    #             out, loss = self.train_step(epoch, i, image_pbar)
    #             pbar.set_description(f'loss: {loss.item():04.2f}')

    #             if terminate:
    #                 print('detecting keyboard interrupt, gracefully exiting')
    #                 return
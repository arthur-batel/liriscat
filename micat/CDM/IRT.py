from IMPACT.model import MIRT
    
        
class CATIRT(MIRT) :

    def __init__(self, **config):
        super().__init__(**config)

    def init_CDM_model(self, train_data: dataset.Dataset, valid_data: dataset.Dataset):
        super().init_model(train_data,valid_data)


        self.params_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['inner_lr'])

        self.params_scaler = torch.amp.GradScaler(self.config['device'])

    def get_params(self):
        return self.model.state_dict()

    def update_params(self,user_ids, question_ids, labels, categories) :
        logging.debug("- Update params : ")
        
        for t in range(self.config['num_inner_epochs']) :

            self.params_optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                loss = self._compute_loss(user_ids, question_ids, categories, labels)

            self.params_scaler.scale(loss).backward()
            self.params_scaler.step(self.params_optimizer)
            self.params_scaler.update()

    def init_test(self, test_data):
        pass
        
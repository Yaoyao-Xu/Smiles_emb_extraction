from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline
import torch
import csv

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = AutoModelWithLMHead.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
tokenizer =  AutoTokenizer.from_pretrained("DeepChem/SmilesTokenizer_PubChem_1M")
extract = pipeline('feature-extraction', model=model, tokenizer=tokenizer)

model = model.to(device)
model.eval()

smiles =[]
smiles_idx = []
with open("rhea-reaction-smiles.tsv") as file:
  tsv_file = csv.reader(file, delimiter="\t")
  for line in tsv_file:
    smiles.append(line[1])
    smiles_idx.append(line[0])
  file.close()

smiles_split = []
# print(len(smiles))

for per_line in smiles:
  molecules = []
  reaction = per_line.split('>>')
  for m in reaction:
    split_by_dot = m.split(".")
    molecules.extend(split_by_dot)
  smiles_split.append(molecules)
    
feactures_dict ={}
extend_max_token_length  = []
with torch.no_grad():
  s = 0  #start point
  for idx , re in enumerate(smiles_split[s:]):
    tensor_list = list()
    for i in range(0, len(re)):
      token_length = len(tokenizer.encode(re[i]))
      if token_length>514:
        extend_max_token_length.append([smiles_idx[idx+s], i])     
      else:
        output = extract(re[i])
        output = torch.tensor(output)
        output = torch.squeeze(output)
        tensor_list.append(output)
    ret  = torch.cat(tensor_list, dim=0)
    # print(ret.shape)
    feactures_dict[smiles_idx[idx+s]] = ret  
    print('smile idx: {}'.format(smiles_idx[idx+s]))
    print('data num#{}'.format(idx+s+1))

        
def list_txt(path, list=None):
    if list != None:
        file = open(path, 'w')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r')
        rdlist = eval(file.read())
        file.close()
        return rdlist

# list_txt(path='D:/collect_features/extend_max.txt', list=extend_max_token_length)
torch.save(feactures_dict, 'Rhea_tensors.pt')




import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import Tuple, Dict
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Email Classifier & AI Response Generator")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class EmailProcessor:
    """Classe para processamento de emails usando NLP"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('portuguese') + stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Palavras-chave para categorização
        self.produtivo_keywords = [
            'reunião', 'projeto', 'trabalho', 'prazo', 'entregável', 
            'cliente', 'apresentação', 'relatório', 'tarefa', 'objetivo',
            'urgente', 'importante', 'prioridade', 'deadline', 'metas',
            'business', 'meeting', 'project', 'work', 'deadline', 'proposta',
            'orçamento', 'contrato', 'negócio', 'desenvolvimento', 'planejamento'
        ]
        
        self.improdutivo_keywords = [
            'spam', 'promoção', 'oferta', 'desconto', 'loteria', 'sorteio',
            'ganhe', 'grátis', 'gratuito', 'assine', 'inscrição', 'ganhou',
            'newsletter', 'marketing', 'publicidade', 'anúncio', 'clique aqui',
            'junk', 'phishing', 'suspeito', 'suspeita', 'fraude', 'herança',
            'promotion', 'discount', 'free', 'win', 'lottery', 'click here'
        ]
    
    def preprocess_text(self, text: str) -> str:
        """Pré-processa o texto do email"""
        if not text:
            return ""
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚâêîôûÂÊÎÔÛàèìòùÀÈÌÒÙãõÃÕçÇ\s]', ' ', text)
        
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_topic(self, text: str) -> str:
        """Extrai o tópico principal do email"""
        topics = {
            'reunião': ['reunião', 'meeting', 'encontro', 'agenda'],
            'projeto': ['projeto', 'project', 'trabalho', 'work'],
            'proposta': ['proposta', 'proposal', 'orçamento', 'quote'],
            'dúvida': ['dúvida', 'question', 'pergunta', 'consulta'],
            'problema': ['problema', 'issue', 'erro', 'bug', 'falha'],
            'relatório': ['relatório', 'report', 'análise', 'dashboard']
        }
        
        text_lower = text.lower()
        for topic, keywords in topics.items():
            if any(keyword in text_lower for keyword in keywords):
                return topic
        
        return "assunto geral"
    
    def predict_category(self, email_text: str) -> Tuple[str, float, Dict]:
        """Classifica o email e retorna análise detalhada"""
        email_lower = email_text.lower()
        
        # Contar keywords
        produtivo_count = sum(1 for word in self.produtivo_keywords if word in email_lower)
        improdutivo_count = sum(1 for word in self.improdutivo_keywords if word in email_lower)
        
        # Palavras encontradas
        produtivo_found = [word for word in self.produtivo_keywords if word in email_lower]
        improdutivo_found = [word for word in self.improdutivo_keywords if word in email_lower]
        
        # Análise de características
        analysis = {
            'produtivo_keywords_found': produtivo_found,
            'improdutivo_keywords_found': improdutivo_found,
            'produtivo_count': produtivo_count,
            'improdutivo_count': improdutivo_count,
            'contains_question': '?' in email_text,
            'contains_urgent': any(word in email_lower for word in ['urgente', 'urgent', 'imediat', 'asap']),
            'contains_deadline': any(word in email_lower for word in ['prazo', 'deadline', 'data limite', 'until']),
            'topic': self.extract_topic(email_text),
            'word_count': len(email_text.split()),
            'has_greeting': any(word in email_lower for word in ['prezado', 'olá', 'bom dia', 'boa tarde', 'caro', 'dear']),
            'has_signature': any(word in email_lower for word in ['atenciosamente', 'sinceramente', 'grato', 'obrigado', 'regards']),
        }
        
        # Determinar categoria
        if improdutivo_count > produtivo_count:
            confidence = min(0.95, 0.5 + (improdutivo_count * 0.15))
            return "Improdutivo", confidence, analysis
        elif produtivo_count > 0:
            confidence = min(0.95, 0.5 + (produtivo_count * 0.1))
            return "Produtivo", confidence, analysis
        else:
            return "Neutro", 0.5, analysis

class AIResponseGenerator:
    """Classe para geração de respostas usando IA"""
    
    def __init__(self):
        self.generator = None
        self.ai_available = False
        self._initialize_generator()
    
    def _initialize_generator(self):
        """Tenta inicializar o gerador de IA de forma segura"""
        try:
            # Tentar importar transformers
            from transformers import pipeline
            
            print("Tentando carregar modelo GPT-2...")
            
            # Usar um modelo mais leve se GPT-2 não estiver disponível
            try:
                self.generator = pipeline(
                    'text-generation',
                    model='gpt2',
                    tokenizer='gpt2',
                    max_length=300,
                    device=-1,  # CPU
                    torch_dtype='auto'
                )
                print("✓ Modelo GPT-2 carregado com sucesso!")
                self.ai_available = True
                
            except Exception as gpt_error:
                print(f"GPT-2 não disponível: {gpt_error}")
                
                # Tentar com um modelo mais leve
                try:
                    print("Tentando carregar modelo distilgpt2...")
                    self.generator = pipeline(
                        'text-generation',
                        model='distilgpt2',
                        tokenizer='distilgpt2',
                        device=-1
                    )
                    print("✓ Modelo distilgpt2 carregado com sucesso!")
                    self.ai_available = True
                    
                except Exception as distil_error:
                    print(f"DistilGPT2 também não disponível: {distil_error}")
                    print("⚠ Usando sistema de templates apenas")
                    self.generator = None
                    self.ai_available = False
        
        except ImportError as import_error:
            print(f"Transformers não instalado: {import_error}")
            print("⚠ Instale: pip install transformers torch")
            self.generator = None
            self.ai_available = False
    
    def generate_response(self, email_text: str, category: str, analysis: Dict) -> str:
        """Gera resposta usando IA ou templates"""
        
        # Primeiro tentar com IA se disponível
        if self.ai_available and self.generator is not None:
            try:
                print("Tentando gerar resposta com IA...")
                ai_response = self._generate_with_ai(email_text, category, analysis)
                if ai_response and len(ai_response.strip()) > 20:
                    print("✓ Resposta gerada com IA")
                    return ai_response
                else:
                    print("⚠ Resposta da IA muito curta, usando template")
            except Exception as e:
                print(f"❌ Erro na geração com IA: {e}")
        
        # Fallback para templates
        print("Usando sistema de templates...")
        return self._generate_template_response(email_text, category, analysis)
    
    def _generate_with_ai(self, email_text: str, category: str, analysis: Dict) -> str:
        """Gera resposta usando modelos de IA"""
        
        # Verificar se o gerador está inicializado
        if self.generator is None:
            raise ValueError("Gerador de IA não inicializado")
        
        # Criar prompt baseado na categoria
        prompt = self._create_prompt(email_text, category, analysis)
        
        try:
            # Configurações para geração
            generation_config = {
                'max_new_tokens': 100,
                'temperature': 0.7,
                'top_p': 0.9,
                'do_sample': True,
                'num_return_sequences': 1,
                'pad_token_id': 50256  # Token de padding para GPT-2
            }
            
            # Gerar resposta
            result = self.generator(prompt, **generation_config)
            
            # Extrair texto gerado
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'generated_text' in result[0]:
                    response = result[0]['generated_text']
                else:
                    response = str(result[0])
            else:
                response = str(result)
            
            # Limpar resposta (remover prompt se necessário)
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            # Remover texto após possíveis marcadores de fim
            for marker in ['\n\n', '---', '===', 'Resposta:']:
                if marker in response:
                    response = response.split(marker)[0].strip()
            
            return response
            
        except Exception as e:
            print(f"Erro durante a geração: {e}")
            raise
    
    def _create_prompt(self, email_text: str, category: str, analysis: Dict) -> str:
        """Cria prompt para a IA"""
        
        topic = analysis.get('topic', 'assunto')
        is_urgent = " (URGENTE)" if analysis.get('contains_urgent') else ""
        has_question = " (contém perguntas)" if analysis.get('contains_question') else ""
        
        if category == "Produtivo":
            return f"""Gere uma resposta profissional em português para este email de trabalho.

Email recebido:
\"\"\"
{email_text[:300]}
\"\"\"

Contexto: {topic}{is_urgent}{has_question}
Categoria: Email Produtivo

Instruções para a resposta:
1. Agradeça pelo email
2. Confirme o recebimento
3. Responda brevemente ao conteúdo
4. Proponha próximos passos
5. Seja profissional e educado
6. Use português formal

Resposta profissional:"""
        
        elif category == "Improdutivo":
            spam_count = analysis.get('improdutivo_count', 0)
            spam_note = f" ({spam_count} indicadores de spam)" if spam_count > 0 else ""
            
            return f"""Gere uma resposta educada mas firme para um email identificado como spam/não solicitado.

Email recebido (suspeito){spam_note}:
\"\"\"
{email_text[:250]}
\"\"\"

Categoria: Email Improdutivo/Spam

Instruções:
1. Agradeça genericamente
2. Indique falta de interesse
3. Sugira remoção da lista (se aplicável)
4. Seja breve e direto
5. Mantenha tom profissional
6. Use português

Resposta:"""
        
        else:
            return f"""Gere uma resposta neutra e profissional para este email.

Email:
\"\"\"
{email_text[:300]}
\"\"\"

Categoria: Email Neutro

Instruções:
1. Agradeça pelo contato
2. Confirme recebimento
3. Ofereça ajuda se necessário
4. Seja breve
5. Use português formal

Resposta:"""
    
    def _generate_template_response(self, email_text: str, category: str, analysis: Dict) -> str:
        """Gera resposta baseada em template"""
        
        topic = analysis.get('topic', 'assunto')
        has_question = analysis.get('contains_question', False)
        is_urgent = analysis.get('contains_urgent', False)
        
        if category == "Produtivo":
            greeting = "Prezados,"
            urgency_note = "\n\nNotamos a urgência mencionada e daremos prioridade ao seu assunto." if is_urgent else ""
            question_note = "\n\nEstamos analisando suas perguntas e retornaremos com respostas em breve." if has_question else ""
            
            response = f"""{greeting}

Agradecemos pelo seu email sobre {topic}.{urgency_note}{question_note}

Confirmamos o recebimento da sua mensagem e estamos processando as informações.

Retornaremos com um posicionamento o mais breve possível.

Atenciosamente,
Equipe de Suporte"""
        
        elif category == "Improdutivo":
            spam_count = analysis.get('improdutivo_count', 0)
            spam_indicator = f" (identificamos {spam_count} indicadores de conteúdo não solicitado)" if spam_count > 0 else ""
            
            response = f"""Olá,

Obrigado pelo seu contato.

No momento, não estamos interessados neste tipo de comunicação{spam_indicator}.

Caso queira ser removido de nossa lista de contatos, por favor responda com "REMOVER" no assunto do email.

Agradecemos a compreensão.

Atenciosamente,
Sistema Automático de Filtragem"""
        
        else:
            response = f"""Prezado remetente,

Agradecemos pelo seu contato. Recebemos sua mensagem sobre {topic}.

Analisaremos o conteúdo e retornaremos caso necessitemos de mais informações.

Atenciosamente,
Equipe"""
        
        return response

# Instanciar processadores
email_processor = EmailProcessor()
ai_generator = AIResponseGenerator()

print(f"Status IA: {'Disponível' if ai_generator.ai_available else 'Não disponível - usando templates'}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Página principal"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "ai_available": ai_generator.ai_available
    })

@app.get("/classificar", response_class=HTMLResponse)
async def classificar_get(request: Request):
    """Endpoint GET para /classificar - redireciona para página principal"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "ai_available": ai_generator.ai_available
    })

@app.post("/classificar", response_class=HTMLResponse)
async def classificar_email(
    request: Request,
    email_text: str = Form(default="")
):
    """Endpoint POST para classificação de email"""
    try:
        if not email_text or not email_text.strip():
            return templates.TemplateResponse(
                "index.html", 
                {
                    "request": request,
                    "error": "Por favor, insira o texto do email.",
                    "email_original": "",
                    "ai_available": ai_generator.ai_available
                }
            )
        
        email_original = email_text.strip()
        print(f"Processando email de {len(email_original)} caracteres...")
        
        # Classificar email
        categoria, confianca, analise = email_processor.predict_category(email_original)
        print(f"Classificado como: {categoria} (confiança: {confianca:.2f})")
        
        # Gerar resposta com IA
        resposta_ia = ai_generator.generate_response(email_original, categoria, analise)
        print(f"Resposta gerada: {len(resposta_ia)} caracteres")
        
        # Processar texto para exibição
        texto_processado = email_processor.preprocess_text(email_original)
        
        # Preparar dados para template
        template_data = {
            "request": request,
            "email_original": email_original,
            "email_processado": texto_processado[:300] + ("..." if len(texto_processado) > 300 else ""),
            "categoria": categoria,
            "confianca": f"{confianca:.1%}",
            "confianca_numero": confianca,
            "resposta_ia": resposta_ia,
            "analise": analise,
            "topic": analise.get('topic', 'Não identificado'),
            "keywords_produtivo": analise.get('produtivo_keywords_found', [])[:5],
            "keywords_improdutivo": analise.get('improdutivo_keywords_found', [])[:5],
            "is_urgent": analise.get('contains_urgent', False),
            "has_question": analise.get('contains_question', False),
            "ai_available": ai_generator.ai_available
        }
        
        return templates.TemplateResponse(
            "resultado.html", 
            template_data
        )
        
    except Exception as e:
        print(f"Erro no processamento: {e}")
        import traceback
        traceback.print_exc()
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"Erro ao processar: {str(e)}",
                "email_original": email_text if email_text else "",
                "ai_available": ai_generator.ai_available
            }
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ai_available": ai_generator.ai_available,
        "processor_ready": True,
        "message": "Sistema de classificação de emails funcionando"
    }

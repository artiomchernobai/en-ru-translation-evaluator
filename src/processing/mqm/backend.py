"""OpenAI-compatible local judge (LM Studio, Ollama, llama.cpp). No ML imports."""
import json
import os
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from .prompt import messages, PROMPT_VERSION
from .errors import MQMRequestError


class LocalMQMBackend:
    def __init__(self, *, model, base_url='http://127.0.0.1:1234/v1', timeout=120, max_tokens=2048,
                 json_mode=True, api_key_env='MQM_API_KEY', max_retries=1):
        url=urlsplit(base_url)
        if url.scheme not in ('http','https') or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('MQM base URL must be HTTP(S) without embedded credentials/query')
        if not isinstance(model,str) or not model.strip():
            raise ValueError('Укажите имя загруженной локальной модели в настройках MQM')
        if type(timeout) not in (int,float) or not 1<=timeout<=3600 or type(max_tokens) is not int or not 128<=max_tokens<=32768:
            raise ValueError('Invalid MQM timeout or output token limit')
        if type(max_retries) is not int or not 0 <= max_retries <= 2:raise ValueError('max_retries must be 0..2')
        self.max_retries=max_retries
        self.url=base_url.rstrip('/')+'/chat/completions'
        self.model,self.timeout,self.max_tokens,self.json_mode=model.strip(),timeout,max_tokens,bool(json_mode)
        self.api_key_env=api_key_env
        self.metadata={'name':self.model,'base_url':base_url.rstrip('/'),'prompt_version':PROMPT_VERSION,
                       'temperature':0,'max_retries':max_retries,'max_output_tokens':max_tokens,'json_mode':self.json_mode,
                       'assessment':'automated_reference_free_mqm_proposals'}

    def judge(self, request):
        body={'model':self.model,'messages':messages(request),'temperature':0,'max_tokens':self.max_tokens,'stream':False}
        if self.json_mode:body['response_format']={'type':'json_object'}
        headers={'Content-Type':'application/json'}
        token=os.environ.get(self.api_key_env)
        if token:headers['Authorization']='Bearer '+token
        try:
            with urlopen(Request(self.url,json.dumps(body,ensure_ascii=False).encode('utf-8'),headers),timeout=self.timeout) as response:
                payload=response.read(4*1024*1024+1)
            if len(payload)>4*1024*1024:raise ValueError('MQM response exceeds size limit')
            answer=json.loads(payload)
            choice=answer['choices'][0]
            if choice.get('finish_reason') not in (None,'stop'):
                raise ValueError('Model response is incomplete (output/context limit or refusal)')
            content=choice['message']['content']
            if not isinstance(content,str):raise ValueError('Model returned no JSON text')
            # Markdown fences are common with local instruction models; no prose/fuzzy repairs.
            content=content.strip()
            if content.startswith('```') and content.endswith('```'):
                content=content.split('\n',1)[1].rsplit('```',1)[0].strip()
            return json.loads(content)
        except HTTPError as error:
            detail=error.read(8192).decode('utf-8',errors='replace')
            try:
                parsed=json.loads(detail)
                value=parsed.get('error',parsed)
                detail=value.get('message',str(value)) if isinstance(value,dict) else str(value)
            except (ValueError,AttributeError):pass
            if token:detail=detail.replace(token,'[REDACTED]')
            detail=' '.join(str(detail).split())[:1500]
            # Retry without optional JSON mode only for an explicit unsupported-format error.
            if (error.code==400 and 'response_format' in body
                    and any(k in detail.lower() for k in ('response_format','json_object'))):
                fallback=type(self)(model=self.model,base_url=self.metadata['base_url'],timeout=self.timeout,
                                    max_tokens=self.max_tokens,json_mode=False,api_key_env=self.api_key_env,max_retries=self.max_retries)
                return fallback.judge(request)
            context_error=error.code in (400,413,422) and any(k in detail.lower() for k in ('context','token','too long','too large'))
            raise MQMRequestError(f'MQM server HTTP {error.code}: {detail or error.reason}',fatal=not context_error) from error
        except (URLError,TimeoutError,OSError) as error:
            raise RuntimeError('Локальный сервер MQM недоступен или превышено время ожидания') from error
        except (KeyError,IndexError,TypeError) as error:
            raise ValueError('Invalid OpenAI-compatible MQM server response') from error

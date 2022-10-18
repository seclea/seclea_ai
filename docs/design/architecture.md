# Asynchronous Architecture

## Discovery

One client will initiate the background server - this is TBD the exact mechanism.
We can use the same as wandb and have a CLI that is called by the integration.
Alternatives are to spawn a process directly - but we need to find a way to separate 
it from the parent process and also make it discoverable to other client threads - in 
multithread/process training context.

Why do we care about one collection/sending point? To ensure consistency of the data
collected.

Major constraints are:
- Sequencing of various entities
  - training runs
  - datasets (parent child relationships)
  - model states 
- Validation 
  - metadata for datasets
  - uniqueness of some items (to avoid rejection by server)
- Consistency
  - must collect all data required and not lose except in major failure
  - ensure flow of information is correct - ie between datasets
- Prevent over requesting portal
  - Use circuit breaker pattern
  - Also use backoff when retrying
- Resource Usage
  - Try not to use too much memory
  - Try not to use too much storage
  - Need to allow user to provide limits on these that will then pause logging until free.
    - RC file?

We will use sqlite3 as a local persistent store. This will store various information about the records being processed
so that we can use them for clean up and ensuring consistency. 

We may also use it for circuit breaker pattern - but we can also use memory I think.

Need to decide whether to use cron or on restart for clean up of incomplete/failed requests.

Some wait for background to complete in the User Thread but should have timeout.

## Data logging process
```mermaid
    sequenceDiagram
  
    participant c as client
    participant b_s as background_server
    participant s as sender
    participant cb as circuit_breaker
    participant b as backend
    
    
    Note over b_s, b: All of this will be async. Not sure on exact structure as dataset verification could change process vs task
    Note over c, b_s: This may be either grpc or REST call.
    c->>b_s: log state or data - serialize data or files
    
    b_s->>c: confirm or too large (pause recording)
    b_s->>s: add to send queue
    s->>cb: buffer and send to backend (async)
    cb->>b: send response
    Note over cb: prevents retrying if server unresponsive etc.
    b->>cb: response
    cb->>s: response
    s->>b_s: return response for handling
```

## Auth sequence
```mermaid
  sequenceDiagram
  
  participant a as AuthServer
  participant c as ClientThread 
  participant i as Internal
  participant s as Sender
  participant db as DB
  participant p as Portal
  
  c->>a: login or get new tokens
  a->>c: return tokens
  
  par
    c->>i: __init__/start (passes auth information)
  and
    c->>db: store tokens
  end
  i->>s: start() pass auth information
  
  s->>p: send data
  p->>s: fail auth
  s->>db: get refresh tokens
  db->>s: return tokens
  s->>a: refresh token
  a->>s: new access token
  s->>p: send data
  p->>s: success
  
```


```mermaid
  classDiagram

  SecleaAI *-- Director
  SecleaAI *-- Api
  SecleaAI *-- DB
  
  AuthService *-- Session
  AuthService *-- DB
  
  Director *-- Writer
  Director *-- Sender
  
  Api *-- AuthService
  Api *-- Session
  
  Writer *-- DB
  
  Sender *-- Api
  Sender *-- DB

  class SecleaAI {
    db: DB
    director: Director
    api: Api
    
    +upload_dataset()
    +upload_training_run()
  }
  

  class AuthService {
    db: DB
    session: Session
    
    +authenticate()
  }
  
  
  class Director {
    store_q: Queue
    send_q: Queue
    db: DB
    writer = Writer
    sender = Sender
    write_threadpool_executor = ThreadPoolExecutor(max_workers=4)
    send_thread_executor = SingleThreadTaskExecutor()
    send_executing: Dict[Future, Dict] = dict()
    write_executing: List[Future] = list()
    errors = list()
    
    +store_entity()
    +send_entity()
    
  }
  
  class Api {
    auth: AuthService
    session: Session
  }
  
  class Writer {
    db: DB
  }
  
  class Sender {
    db: DB
    api: Api
  }
  
  class DB {
  
  }
  
```


## Persistence classes (internal)
These need to be serializable to JSON. Will need to deal with NaNs.
All the Record subclasses are optional - we can start with more generic Record and add subclasses
if they make sense for validation etc.
```mermaid
    classDiagram

    Entity <|-- Dataset
    Entity <|-- ModelState
    Entity <|-- TrainingRun
    Entity <|-- DatasetTransformation
    Entity <|-- Model
    Entity <|-- Project
    Record *-- Entity

    class Record {
        record: Record
        status: Saving | Transmitting | Failed | Completed
        error: str
        retries: int 
    }

    class Entity {
        remote_id: str
        name: str
        file_path: str
        size: int
    }

    class Dataset {
        file_path: str
        size: int
        intermediate: bool
        metadata: dict (to str for sqlite)
    }

    class ModelState {
        id: int (autogen)
        remote_id: str
        file_path: str
        size: int
        metadata: dict (to str for sqlite)
        
    }

    class DatasetTransformation {
      id: int (autogen)
      remote_id: str
      name: str
      dataset: Dataset
    }

    class TrainingRun {
      id: int (autogen)
      remote_id: str
      name: str
    }

    class Model {
      id: int (autogen)
      remote_id: str
      name: str
      framework: str?
    }

    class Project {
      id: int (autogen)
      remote_id: str
      name: str
    }
```

## External classes
We will also have classes for Users to use in their code. These will be added here.

## Error Handling

### Error modes:
- Run out of space/memory - later
- Request failures - later
  - API
  - Auth
- DB errors
- Saving errors
  - save_model_state
- Threading errors
  - ~~fail to start~~
  - ~~error in parent - children need to exit~~
  - queue errors
    - encoding
    - full?
  - error during operation
    - saving
      - os issues
    - API
      - fail to auth
      - various error responses
    - DB
      - fail to connect
      - deadlock
  - error on complete
    - queue not empty
    - not sent
    - partial send?


### Network failure modes

- 400 Bad Request - There is probably a mismatch with acceptable data types or wrong format out of date]
- 401
- 402
- 403 Unauthorized
- 404 Not found
- 405
- timeout
- 500 Internal Server Error

```mermaid
    flowchart TD
    
    400
    
    401
    402
    
    r4[response status == 403] --> auth[reauth or wait]
    
    404
    
    timeout.
    
    r5[response status == 500] --> re[wait - with backoff]
    
```

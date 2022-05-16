# Asynchronous Architecture

## Discovery

One client will initiate the background server - this is TBD the exact mechanism.
We can use the same as wandb and have a CLI that is called by the integration.
Alternatives are to spawn a process directly - but we need to find a way to separate 
it from the parent process and also make it discoverable to other client threads - in 
multithread/process training context.

Why do we care about one collection/sending point? To ensure consistency of the data
collected.


## Data logging process
```mermaid
    sequenceDiagram
  
    participant c as client
    participant b_s as background_server
    participant s as sender
    participant b as backend
    
    
    Note over b_s, b: All of this will be async. Not sure on exact structure as dataset verification could change process vs task
    Note over c, b_s: This may be either grpc or REST call.
    c->>b_s: log state or data - serialize data or files
    
    b_s->>c: confirm
    b_s->>s: add to send queue
    s->>b: buffer and send to backend (async)
    b->>s: response
    s->>b_s: return response for handling
```

## Class Diagram
```mermaid
    classDiagram

    class BackgroundServer {
        - Director
        + start()
        + stop()
    }



```
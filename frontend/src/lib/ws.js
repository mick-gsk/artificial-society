// Thin live-frame WebSocket client with auto-reconnect.
//
// The server sends one `hello` message (the stable biome + behaviour legends)
// on connect, then a stream of `frame` messages (~20 Hz) deduplicated by tick.
// In dev the `/ws` path is proxied to uvicorn by Vite; in prod it is same-origin.
//
// The client can talk back over the socket via the returned `send` — the only
// message the server understands today is `{type:"inspect", id:<int|null>}`.
// ws.js keeps NO inspect state of its own: the owner (World.svelte) re-sends its
// current inspect after every `onHello` so an auto-reconnect re-subscribes.

export function connectWS({ onHello, onFrame, onOpen, onClose } = {}) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${location.host}/ws`;
  let sock = null;
  let closed = false;
  let retry = null;

  function open() {
    sock = new WebSocket(url);
    sock.onopen = () => onOpen?.();
    sock.onmessage = (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.type === "hello") onHello?.(msg);
      else if (msg.type === "frame") onFrame?.(msg);
    };
    sock.onclose = () => {
      onClose?.();
      if (!closed) retry = setTimeout(open, 1000);
    };
    sock.onerror = () => sock && sock.close();
  }

  open();

  // Fire-and-forget send; JSON-encodes and only writes on an OPEN socket, so a
  // call during (re)connect or teardown is a harmless no-op that returns false.
  function send(obj) {
    if (!sock || sock.readyState !== WebSocket.OPEN) return false;
    try {
      sock.send(JSON.stringify(obj));
      return true;
    } catch {
      return false;
    }
  }

  // Disposer: stop reconnecting and close the socket.
  function dispose() {
    closed = true;
    if (retry) clearTimeout(retry);
    if (sock) sock.close();
  }

  return { dispose, send };
}

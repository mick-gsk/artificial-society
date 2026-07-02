// Thin live-frame WebSocket client with auto-reconnect.
//
// The server sends one `hello` message (the stable biome legend) on connect,
// then a stream of `frame` messages (~20 Hz) deduplicated by tick. In dev the
// `/ws` path is proxied to uvicorn by Vite; in prod it is same-origin.

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

  // Disposer: stop reconnecting and close the socket.
  return () => {
    closed = true;
    if (retry) clearTimeout(retry);
    if (sock) sock.close();
  };
}

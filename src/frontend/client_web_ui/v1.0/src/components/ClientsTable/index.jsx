import Client from "../Client";
import React from "react";

const ClientTable = ({clients}) => {

    return (
      <>
        <div className="sections-list divide-y divide-slate-200">
          {(
              clients.map((client) => (
                <Client key={client.id} client={client} />
              ))
          )}
        </div>
      </>
    )
}

export default ClientTable;